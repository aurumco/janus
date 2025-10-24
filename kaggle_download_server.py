"""Kaggle download server with ngrok integration."""

import sys
import argparse
import threading
import http.server
import socketserver
from pathlib import Path
from typing import List


def format_size(size_bytes: int) -> str:
    """Format bytes to human readable."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def zip_directories(
    output_zip: str,
    directories: List[str],
    base_dir: str = "/kaggle/working"
) -> bool:
    """Zip directories into a single archive."""
    
    import zipfile
    
    zip_path = Path(base_dir) / output_zip
    
    print(f"Creating archive: {zip_path.name}")
    
    valid_dirs = []
    for dir_path in directories:
        full_path = Path(dir_path)
        if not full_path.is_absolute():
            full_path = Path(base_dir) / dir_path
        
        if full_path.exists():
            valid_dirs.append(full_path)
            print(f"  - adding: {full_path}")
        else:
            print(f"  - skipped (not found): {full_path}")
    
    if not valid_dirs:
        print("error: no valid directories to zip")
        return False
    
    try:
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for dir_path in valid_dirs:
                if dir_path.is_file():
                    arcname = dir_path.name
                    zipf.write(dir_path, arcname=arcname)
                else:
                    for file_path in dir_path.rglob('*'):
                        if file_path.is_file():
                            arcname = file_path.relative_to(dir_path.parent)
                            zipf.write(file_path, arcname=arcname)
        
        size = zip_path.stat().st_size
        print(f"✓ Archive created: {format_size(size)}")
        return True
    
    except Exception as e:
        print(f"error: failed to create archive: {e}")
        return False


def start_download_server(
    file_path: str,
    ngrok_token: str = "34DiKV7lbe4YNTJLY2N4spWyps0_2XtEkCauoJUftbLB4xPG3",
    port: int = 0
) -> None:
    """Start HTTP server with ngrok tunnel."""
    
    file_path = Path(file_path)
    
    if not file_path.exists():
        print(f"error: file not found: {file_path}")
        sys.exit(1)
    
    file_size = file_path.stat().st_size
    print(f"\nFile ready: {file_path.name}")
    print(f"- size: {format_size(file_size)}")
    
    try:
        from pyngrok import ngrok
    except ImportError:
        print("error: pyngrok not installed")
        print("install: !pip install pyngrok")
        sys.exit(1)
    
    directory = str(file_path.parent.absolute())
    filename = file_path.name
    
    print(f"\nConfiguring ngrok...")
    try:
        ngrok.set_auth_token(ngrok_token)
        print("✓ ngrok authenticated")
    except Exception as e:
        print(f"error: ngrok auth failed: {e}")
        sys.exit(1)
    
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=directory, **kwargs)
        
        def log_message(self, format, *args):
            pass
    
    try:
        httpd = socketserver.TCPServer(("", port), Handler)
        actual_port = httpd.server_address[1]
        print(f"✓ HTTP server bound to port {actual_port}")
        
        server_thread = threading.Thread(target=httpd.serve_forever)
        server_thread.daemon = True
        server_thread.start()
        
        public_url = ngrok.connect(actual_port)
        
        print("\n" + "="*70)
        print("DOWNLOAD LINK:")
        print(f"{public_url}/{filename}")
        print("="*70)
        print("\nCopy the link above into your browser to download.")
        print("File size:", format_size(file_size))
        print("\nIMPORTANT: After download completes, manually STOP this cell.")
        print("="*70)
        
        input("\nPress Enter to stop server...")
        
    except KeyboardInterrupt:
        print("\nShutting down server...")
    except Exception as e:
        print(f"error: {e}")
    finally:
        if 'httpd' in locals():
            httpd.shutdown()
        ngrok.kill()


def main():
    parser = argparse.ArgumentParser(
        description='Kaggle Download Server',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Zip and serve dataset
  python kaggle_download_server.py --mode dataset --output janus_dataset.zip
  
  # Zip and serve model checkpoints
  python kaggle_download_server.py --mode model --output janus_model.zip
  
  # Zip custom directories
  python kaggle_download_server.py --mode custom --dirs outputs logs --output janus_custom.zip
  
  # Serve existing file
  python kaggle_download_server.py --serve /kaggle/working/existing.zip
        """
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['dataset', 'model', 'both', 'custom'],
        help='What to zip: dataset, model, both, or custom'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='janus_archive.zip',
        help='Output zip filename (default: janus_archive.zip)'
    )
    parser.add_argument(
        '--dirs',
        nargs='+',
        help='Custom directories to zip (for --mode custom)'
    )
    parser.add_argument(
        '--base-dir',
        type=str,
        default='/kaggle/working',
        help='Base directory (default: /kaggle/working)'
    )
    parser.add_argument(
        '--serve',
        type=str,
        help='Serve existing file without zipping'
    )
    parser.add_argument(
        '--ngrok-token',
        type=str,
        default='34DiKV7lbe4YNTJLY2N4spWyps0_2XtEkCauoJUftbLB4xPG3',
        help='ngrok auth token'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=0,
        help='HTTP server port (default: 0 for random)'
    )
    
    args = parser.parse_args()
    
    if args.serve:
        start_download_server(
            file_path=args.serve,
            ngrok_token=args.ngrok_token,
            port=args.port
        )
        return
    
    if not args.mode:
        print("error: --mode or --serve required")
        parser.print_help()
        sys.exit(1)
    
    directories = []
    
    if args.mode == 'dataset':
        directories = [
            'outputs/datasets',
            'janus/outputs/datasets'
        ]
    elif args.mode == 'model':
        directories = [
            'results',
            'logs',
            'checkpoints',
            'outputs/models'
        ]
    elif args.mode == 'both':
        directories = [
            'outputs',
            'results',
            'logs',
            'checkpoints'
        ]
    elif args.mode == 'custom':
        if not args.dirs:
            print("error: --dirs required for custom mode")
            sys.exit(1)
        directories = args.dirs
    
    output_path = Path(args.base_dir) / args.output
    
    success = zip_directories(
        output_zip=args.output,
        directories=directories,
        base_dir=args.base_dir
    )
    
    if not success:
        sys.exit(1)
    
    start_download_server(
        file_path=str(output_path),
        ngrok_token=args.ngrok_token,
        port=args.port
    )


if __name__ == '__main__':
    main()
