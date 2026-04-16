import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from typing import Optional, Tuple

class NoHiddenLayerNN(nn.Module):
    """
    A simple neural network architecture with no hidden layers, functionally 
    equivalent to linear or logistic regression depending on the activation function.
    """
    def __init__(
        self, 
        input_size: int, 
        output_size: int, 
        activation_fn: Optional[nn.Module] = None
    ) -> None:
        """
        Initializes the zero-hidden-layer neural network.

        Args:
            input_size (int): The number of input features.
            output_size (int): The number of target outputs.
            activation_fn (nn.Module, optional): The activation function applied 
                to the output layer. Defaults to nn.Identity() (linear output).
        """
        super().__init__()

        if activation_fn is None:
            activation_fn = nn.Identity()
        
        # Flatten input to ensure compatibility with fully connected layers
        self.flatten = nn.Flatten() 
        
        # Define the network architecture map
        self.net = nn.Sequential(
            nn.Linear(input_size, output_size),
            activation_fn
        )
        
        # Define default training hyperparameters
        self.loss_fn = nn.MSELoss()
        self.lr = 0.01
        self.maxepochs = 400
        self.batch_size = 32
        
        # Early stopping constraints to prevent overfitting and save compute
        self.patience = 25      
        self.min_delta = 1e-4   

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Executes a forward pass through the network.

        Args:
            x (torch.Tensor): The input feature tensor.

        Returns:
            torch.Tensor: The predicted logits or activated outputs.
        """
        x = self.flatten(x)
        logits = self.net(x)
        return logits

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        """
        Executes the training loop using mini-batch gradient descent and early stopping.

        Args:
            X (torch.Tensor): The training dataset features.
            y (torch.Tensor): The training dataset targets.
        """
        # Enable gradient tracking and layer updates
        super().train()

        # Initialize the optimizer to manage weight updates
        optimizer = optim.SGD(self.parameters(), lr=self.lr)

        # Bundle features and targets into a unified dataset for batched iteration
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True) 

        print(f"Starting training with batch size {self.batch_size}...")

        # --- EARLY STOPPING TRACKERS ---
        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.maxepochs):
            epoch_loss = 0.0 
            
            # --- THE MINI-BATCH LOOP ---
            for batch_X, batch_y in dataloader:
                # 1. Forward Pass: Compute predictions
                predictions = self.forward(batch_X)
                
                # 2. Loss Calculation: Quantify prediction error
                loss = self.loss_fn(predictions, batch_y)
                
                # 3. Backward Pass: Flush old gradients, compute new ones, and update weights
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() 
            
            # Normalize epoch loss by the number of batches to get the true average
            avg_loss = epoch_loss / len(dataloader) 

            # Periodically broadcast training progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.maxepochs} | Avg Loss: {avg_loss:.4g}")

            # --- EARLY STOPPING LOGIC ---
            # Assess if the model has yielded a statistically significant improvement
            if best_loss - avg_loss > self.min_delta:
                best_loss = avg_loss
                patience_counter = 0  # Reset tolerance tracker
            else:
                patience_counter += 1 # Penalize for stagnation

            # Terminate training prematurely if stagnation exceeds tolerance
            if patience_counter >= self.patience:
                print(f"\nEarly stopping triggered at Epoch {epoch+1}!")
                print(f"Loss hasn't improved by more than {self.min_delta} for {self.patience} consecutive epochs.")
                break

        print("\nTraining complete!")
    
    def predict(self, X: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Evaluates the trained model on an unseen validation or test dataset.

        Args:
            X (torch.Tensor): The testing dataset features.
            y (torch.Tensor): The testing dataset targets.

        Returns:
            Tuple[torch.Tensor, float]: A tuple containing the raw tensor predictions 
                and the final calculated loss value.
        """
        # Lock network weights and disable dropout/batchnorm for deterministic inference
        self.eval() 
        
        # Suspend computational graph tracking to conserve memory during evaluation
        with torch.no_grad(): 
            
            predictions = self.forward(X) 
            test_loss = self.loss_fn(predictions, y)
            
        print(f"Test Loss: {test_loss.item():.4g}")
        
        # Revert back to training mode as a safety precaution for subsequent operations
        super().train() 
        
        return (predictions, test_loss.item())


class OneHiddenLayerNN(nn.Module):
    """
    A multi-layer perceptron (MLP) architecture containing exactly one hidden layer.
    """
    def __init__(
        self, 
        input_size: int, 
        hidden_size: int, 
        output_size: int, 
        hidden_activation_fn: Optional[nn.Module] = None, 
        output_activation_fn: Optional[nn.Module] = None
    ) -> None:
        """
        Initializes the single-hidden-layer neural network.

        Args:
            input_size (int): The number of input features.
            hidden_size (int): The number of neurons in the hidden layer.
            output_size (int): The number of target outputs.
            hidden_activation_fn (nn.Module, optional): Activation for the hidden layer. Defaults to nn.Sigmoid().
            output_activation_fn (nn.Module, optional): Activation for the output layer. Defaults to nn.Identity().
        """
        super().__init__()

        if hidden_activation_fn is None:
            hidden_activation_fn = nn.Sigmoid()
        if output_activation_fn is None:
            output_activation_fn = nn.Identity()

        # Flatten input to ensure compatibility with fully connected layers
        self.flatten = nn.Flatten() 
        
        # Define the network architecture map
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            hidden_activation_fn,
            nn.Linear(hidden_size, output_size),
            output_activation_fn
        )
        
        # Define default training hyperparameters
        self.loss_fn = nn.MSELoss()
        self.lr = 0.01
        self.maxepochs = 400
        self.batch_size = 32
        
        # Early stopping constraints
        self.patience = 25      
        self.min_delta = 1e-4   

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Executes a forward pass through the network.

        Args:
            x (torch.Tensor): The input feature tensor.

        Returns:
            torch.Tensor: The predicted logits or activated outputs.
        """
        x = self.flatten(x)
        logits = self.net(x)
        return logits

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        """
        Executes the training loop using mini-batch gradient descent and early stopping.

        Args:
            X (torch.Tensor): The training dataset features.
            y (torch.Tensor): The training dataset targets.
        """
        # Enable gradient tracking and layer updates
        super().train()

        # Initialize the optimizer to manage weight updates
        optimizer = optim.SGD(self.parameters(), lr=self.lr)

        # Bundle features and targets into a unified dataset for batched iteration
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True) 

        print(f"Starting training with batch size {self.batch_size}...")

        # --- EARLY STOPPING TRACKERS ---
        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.maxepochs):
            epoch_loss = 0.0 
            
            # --- THE MINI-BATCH LOOP ---
            for batch_X, batch_y in dataloader:
                # 1. Forward Pass
                predictions = self.forward(batch_X)
                
                # 2. Loss Calculation
                loss = self.loss_fn(predictions, batch_y)
                
                # 3. Backward Pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() 
            
            # Normalize epoch loss
            avg_loss = epoch_loss / len(dataloader) 

            # Periodically broadcast training progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.maxepochs} | Avg Loss: {avg_loss:.4g}")

            # --- EARLY STOPPING LOGIC ---
            # Assess if the model has yielded a statistically significant improvement
            if best_loss - avg_loss > self.min_delta:
                best_loss = avg_loss
                patience_counter = 0  # Reset tolerance tracker
            else:
                patience_counter += 1 # Penalize for stagnation

            # Terminate training prematurely if stagnation exceeds tolerance
            if patience_counter >= self.patience:
                print(f"\nEarly stopping triggered at Epoch {epoch+1}!")
                print(f"Loss hasn't improved by more than {self.min_delta} for {self.patience} consecutive epochs.")
                break

        print("\nTraining complete!")
    
    def predict(self, X: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Evaluates the trained model on an unseen validation or test dataset.

        Args:
            X (torch.Tensor): The testing dataset features.
            y (torch.Tensor): The testing dataset targets.

        Returns:
            Tuple[torch.Tensor, float]: A tuple containing the raw tensor predictions 
                and the final calculated loss value.
        """
        # Lock network weights and disable dropout/batchnorm for deterministic inference
        self.eval() 
        
        # Suspend computational graph tracking to conserve memory during evaluation
        with torch.no_grad(): 
            predictions = self.forward(X) 
            test_loss = self.loss_fn(predictions, y)
            
        print(f"Test Loss: {test_loss.item():.4g}")
        
        # Revert back to training mode
        super().train() 
        
        return (predictions, test_loss.item())


class TwoHiddenLayerNN(nn.Module):
    """
    A multi-layer perceptron (MLP) architecture containing exactly two hidden layers.
    """
    def __init__(
        self, 
        input_size: int, 
        hidden_size_1: int, 
        hidden_size_2: int, 
        output_size: int, 
        hidden_activation_fn_1: Optional[nn.Module] = None, 
        hidden_activation_fn_2: Optional[nn.Module] = None, 
        output_activation_fn: Optional[nn.Module] = None
    ) -> None:
        """
        Initializes the two-hidden-layer neural network.

        Args:
            input_size (int): The number of input features.
            hidden_size_1 (int): The number of neurons in the first hidden layer.
            hidden_size_2 (int): The number of neurons in the second hidden layer.
            output_size (int): The number of target outputs.
            hidden_activation_fn_1 (nn.Module, optional): Activation for first hidden layer. Defaults to nn.Sigmoid().
            hidden_activation_fn_2 (nn.Module, optional): Activation for second hidden layer. Defaults to nn.Sigmoid().
            output_activation_fn (nn.Module, optional): Activation for the output layer. Defaults to nn.Identity().
        """
        super().__init__()

        if hidden_activation_fn_1 is None:
            hidden_activation_fn_1 = nn.Sigmoid()
        if hidden_activation_fn_2 is None:
            hidden_activation_fn_2 = nn.Sigmoid()
        if output_activation_fn is None:
            output_activation_fn = nn.Identity()
        
        # Flatten input to ensure compatibility with fully connected layers
        self.flatten = nn.Flatten() 
        
        # Define the network architecture map
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size_1),
            hidden_activation_fn_1,
            nn.Linear(hidden_size_1, hidden_size_2),
            hidden_activation_fn_2,
            nn.Linear(hidden_size_2, output_size),
            output_activation_fn
        )
        
        # Define default training hyperparameters
        self.loss_fn = nn.MSELoss()
        self.lr = 0.01
        self.maxepochs = 400
        self.batch_size = 32
        
        # Early stopping constraints
        self.patience = 25      
        self.min_delta = 1e-4   

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Executes a forward pass through the network.

        Args:
            x (torch.Tensor): The input feature tensor.

        Returns:
            torch.Tensor: The predicted logits or activated outputs.
        """
        x = self.flatten(x)
        logits = self.net(x)
        return logits

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        """
        Executes the training loop using mini-batch gradient descent and early stopping.

        Args:
            X (torch.Tensor): The training dataset features.
            y (torch.Tensor): The training dataset targets.
        """
        # Enable gradient tracking and layer updates
        super().train()

        # Initialize the optimizer to manage weight updates
        optimizer = optim.SGD(self.parameters(), lr=self.lr)

        # Bundle features and targets into a unified dataset for batched iteration
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True) 

        print(f"Starting training with batch size {self.batch_size}...")

        # --- EARLY STOPPING TRACKERS ---
        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.maxepochs):
            epoch_loss = 0.0 
            
            # --- THE MINI-BATCH LOOP ---
            for batch_X, batch_y in dataloader:
                # 1. Forward Pass
                predictions = self.forward(batch_X)
                
                # 2. Loss Calculation
                loss = self.loss_fn(predictions, batch_y)
                
                # 3. Backward Pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() 
            
            # Normalize epoch loss
            avg_loss = epoch_loss / len(dataloader) 

            # Periodically broadcast training progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.maxepochs} | Avg Loss: {avg_loss:.4g}")

            # --- EARLY STOPPING LOGIC ---
            # Assess if the model has yielded a statistically significant improvement
            if best_loss - avg_loss > self.min_delta:
                best_loss = avg_loss
                patience_counter = 0  # Reset tolerance tracker
            else:
                patience_counter += 1 # Penalize for stagnation

            # Terminate training prematurely if stagnation exceeds tolerance
            if patience_counter >= self.patience:
                print(f"\nEarly stopping triggered at Epoch {epoch+1}!")
                print(f"Loss hasn't improved by more than {self.min_delta} for {self.patience} consecutive epochs.")
                break

        print("\nTraining complete!")
    
    def predict(self, X: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Evaluates the trained model on an unseen validation or test dataset.

        Args:
            X (torch.Tensor): The testing dataset features.
            y (torch.Tensor): The testing dataset targets.

        Returns:
            Tuple[torch.Tensor, float]: A tuple containing the raw tensor predictions 
                and the final calculated loss value.
        """
        # Lock network weights and disable dropout/batchnorm for deterministic inference
        self.eval() 
        
        # Suspend computational graph tracking to conserve memory during evaluation
        with torch.no_grad(): 
            predictions = self.forward(X) 
            test_loss = self.loss_fn(predictions, y)
            
        print(f"Test Loss: {test_loss.item():.4g}")
        
        # Revert back to training mode
        super().train() 
        
        return (predictions, test_loss.item())