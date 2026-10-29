import os
import torch
import time

def check_progress(ckpt_dir='./checkpoints'):
    best_ckpt_path = os.path.join(ckpt_dir, 'aodnet_best.pth')
    
    if not os.path.exists(best_ckpt_path):
        print("Training hasn't saved a checkpoint yet. Please wait...")
        return
        
    try:
        # Load the checkpoint to read metadata
        checkpoint = torch.load(best_ckpt_path, map_location='cpu', weights_only=False)
        epoch = checkpoint.get('epoch', 'Unknown')
        best_psnr = checkpoint.get('best_psnr', 'Unknown')
        
        # Get the time the checkpoint was last updated
        last_modified = os.path.getmtime(best_ckpt_path)
        last_modified_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(last_modified))
        
        print(f"--- Live Training Progress ---")
        print(f"Latest Checkpoint Updated : {last_modified_str}")
        print(f"Current Best Epoch        : {epoch}")
        
        if isinstance(best_psnr, (int, float)):
            print(f"Best PSNR Achieved        : {best_psnr:.2f} dB")
        else:
            print(f"Best PSNR Achieved        : {best_psnr}")
            
        print(f"------------------------------")
    except Exception as e:
        print(f"Could not read checkpoint right now (it might be actively writing). Retrying shortly... (Error: {e})")

if __name__ == "__main__":
    try:
        while True:
            # Clear the terminal for a clean live-updating view
            os.system('cls' if os.name == 'nt' else 'clear')
            
            check_progress()
            
            print("\nMonitoring training... Press Ctrl+C to exit.")
            print("Refreshing automatically in 5 seconds...")
            time.sleep(5)
            
    except KeyboardInterrupt:
        print("\nStopped monitoring.")
