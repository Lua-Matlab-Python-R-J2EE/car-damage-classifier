import torch
import torch.nn as nn
from torchvision import transforms
import copy 
import config
#=====================================================================================
# Define these for ImageNet normalization
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]
#=====================================================================================
# write val_model()
def val_model(*,
              model:      torch.nn.Module,
              val_loader: torch.utils.data.DataLoader,
              loss_fn:    torch.nn.Module
              ):
    
    model.eval()
    val_loss, correct, total = 0.0, 0, 0    

    # Create empty containers to accumulate all samples
    all_predictions = []
    all_labels = []

    with torch.inference_mode():
        for i, batch in enumerate(val_loader):
            images, labels = batch['image'].to(device), batch['label'].to(device)
            
            outputs     = model(images)
            loss        = loss_fn(outputs, labels)
            val_loss   += loss.item() * images.size(0)

            predicted   = torch.argmax(outputs, dim=1)
            total      += labels.size(0)
            correct    += (predicted == labels).sum().item() 
            
            # Move data back to CPU and convert to numpy arrays before extending lists
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    avg_loss_per_img = val_loss / len(val_loader.dataset)
    accuracy = 100 * correct / total
            
    # Return the metrics AND the raw data needed for scikit-learn
    return accuracy, avg_loss_per_img, all_predictions, all_labels

#=======================================================================================
# write train_model()
# NOTE: loss.item() = already averaged per image in batch
#       Dividing epoch_loss by len(train_loader) gives average loss per image across full epoch.
def train_model(*,
                epochs:       int,
                model:        torch.nn.Module,
                train_loader: torch.utils.data.DataLoader,
                val_loader:   torch.utils.data.DataLoader,
                loss_fn:      torch.nn.Module,
                optimizer:    torch.optim.Optimizer, 
                scheduler:    torch.optim.lr_scheduler.ReduceLROnPlateau
               ) -> dict:  # CHANGED: Now returns a dictionary of weights

    model.train()
    # --- NEW FIX: Tracking variables for the best model ---
    best_val_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    
    for epoch in range(epochs):
        # NEW FIXED: Explicitly force model back to train mode at the start of every epoch
        model.train()
        epoch_loss = 0.0
        
        for i, batch in enumerate(train_loader):
            images, labels = batch['image'].to(device), batch['label'].to(device)

            # forward pass
            outputs = model(images)
            loss    = loss_fn(outputs, labels)
            
            # backward pass
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            epoch_loss += loss.item() * images.size(0) 
            
            # step 
            optimizer.step()

            # print after 10 iterations of i
            if i%10 == 0:
                print(f"---- Epoch [{epoch+1}/{epochs}], Batch [{i+1}/{len(train_loader)}], Batch_Loss {loss.item():.3f} ")

        # after the batch loop ends, and before next epoch, get validation metric
        val_acc, val_loss,_,_ = val_model(model=model,
                                     val_loader=val_loader, 
                                     loss_fn=loss_fn)
        
        # --- ADDED: Check and save the best weights (This triggers at Epoch 7 in the best case) ---
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            print(f"--> New best accuracy achieved: {val_acc:.2f}%. Saving checkpoint weights.")
        
        # update scheduler based on val_loss, and reduce step size if loss is not decreasing
        scheduler.step(val_loss)

        # Get lr value from optimizer for printing
        current_lr = optimizer.param_groups[0]['lr'] 
        
        # Print epoch, val_acc, val_loss
        avg_train_loss_per_img = epoch_loss / len(train_loader.dataset)  
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss_per_img:.3f}, Val Acc: {val_acc:.3f}%, Val Loss: {val_loss:.3f}, LR: {current_lr}")

    # --- ADDED: Return the best parameters after all epochs finish ---
    return best_model_wts       

#=======================================================================================
class CarClassifierCNN(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes, dropout_p):
        super().__init__()
        
        self.num_classes = num_classes
        
        # BatchNorm in conv layers -> no Dropout needed as Dropout belongs after Flatten in the classifier
        # Conv2d + BatchNorm2d -> always set bias=False, bec BatchNorm does this internally:
        # Step 1: Normalize  => x = (x - mean) / std
        # Step 2: Scale      => x = gamma * x + beta   (beta acts as bias)!
        
        # Conv1
        self.feature1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False), # 32 x 16 x 224 x 224
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 32 x 16 x 112 x 112
        )      
        
        # Conv2
        self.feature2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=int(out_channels*2), kernel_size=3, stride=1, padding=1, bias=False), # 32 x 32 x 112 x 112
            nn.BatchNorm2d(int(out_channels*2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 32 x 32 x 56 x 56
        )     

        # Conv3
        self.feature3 = nn.Sequential(
            nn.Conv2d(in_channels=int(out_channels*2), out_channels=int(out_channels*4), kernel_size=3, stride=1, padding=1, bias=False), # 32 x 64 x 56 x 56
            nn.BatchNorm2d(int(out_channels*4)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 32 x 64 x 28 x 28
        )      
        
        # Conv4
        self.feature4 = nn.Sequential(
            nn.Conv2d(in_channels=int(out_channels*4), out_channels=int(out_channels*8), kernel_size=3, stride=1, padding=1, bias=False), # 32 x 128 x 28 x 28
            nn.BatchNorm2d(int(out_channels*8)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 32 x 128 x 14 x 14
        )

        # Adaptive pool
        self.pool = nn.AdaptiveAvgPool2d(1) # 32 x 128 x 1 x 1
        
        # Flattening
        self.flatten = nn.Flatten()

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear( in_features=int(out_channels*8), out_features=int(out_channels*8*2) ),
            nn.ReLU(),
            nn.Dropout( p=dropout_p ),
            nn.Linear( in_features=int(out_channels*8*2), out_features=num_classes )
        )
        
    def forward(self, x):
        x = self.feature1(x)
        x = self.feature2(x)
        x = self.feature3(x)
        x = self.feature4(x)
        x = self.pool(x)      # batch x features x 1 x 1]
        x = self.flatten(x)   # batch x features
        x = self.classifier(x)            
        return x            
#=======================================================================================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),            
    transforms.RandomHorizontalFlip(),    
    transforms.RandomRotation(10),        
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
])
#=======================================================================================
val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),    
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
])
#=======================================================================================
# need wrapper fn to apply train_transform as with_transform passes data as a batch dictionary
# batch attributes match the dataset features. 
# i.e. whatever ds['train'].features shows => those are the batch keys
# NOTE: This function process one image at a time
def apply_train_transform(batch):
    # NEW FIX: Added fallback checking to ensure objects convert to RGB cleanly
    batch['image'] = [train_transform(img.convert('RGB') if hasattr(img, 'convert') else img) for img in batch['image']]
    return batch
    
#=======================================================================================
# # Apply transformation on test, val, train sets seperately
def apply_val_test_transform(batch):
    batch['image'] = [val_test_transform(img.convert('RGB') if hasattr(img, 'convert') else img) for img in batch['image']]
    return batch
#=======================================================================================
def get_min_image_dimensions(dataset_split):
    """
    Finds the minimum width and height from a Hugging Face dataset split
    containing PIL images, using a single-pass loop for memory efficiency.
    """
    # Ensure iteration yields Python/PIL objects
    stream = dataset_split.with_format("python")
    
    # Initialize with infinity so any real size will be smaller
    min_w = float('inf')
    min_h = float('inf')
    
    # Single pass loop avoids loading the dataset twice
    for row in stream:
        w, h = row["image"].size
        if w < min_w:
            min_w = w
        if h < min_h:
            min_h = h
            
    return min_w, min_h
#=======================================================================================
