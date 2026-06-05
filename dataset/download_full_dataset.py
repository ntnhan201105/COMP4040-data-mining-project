"""Script to download the complete ACN-Data dataset using the acnportal API."""
import argparse
import json
import sys
import datetime
from pathlib import Path

from acnportal import acndata
from tqdm import tqdm


class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle datetime objects returned by DataClient."""
    def default(self, obj):
        if isinstance(obj, datetime.datetime):
            # Format expected by our event_loader.py
            return obj.strftime("%a, %d %b %Y %H:%M:%S GMT")
        return super().default(obj)


def download_site_data(api_token: str, site: str, output_path: Path):
    print(f"\n--- Downloading data for site: {site.upper()} ---")
    try:
        client = acndata.DataClient(api_token)
    except Exception as e:
        print(f"Failed to authenticate with ACN-Data API: {e}")
        print("Please ensure your API token is correct.")
        sys.exit(1)

    print(f"Connected to API. Fetching sessions for {site}...")
    print("This may take several minutes as there are tens of thousands of sessions.")
    
    sessions = []
    try:
        # get_sessions returns a generator yielding session dictionaries
        for session in tqdm(client.get_sessions(site), desc=f"Downloading {site}"):
            sessions.append(session)
    except Exception as e:
        print(f"\nError while fetching data: {e}")
        if len(sessions) == 0:
            sys.exit(1)
        print(f"Managed to download {len(sessions)} sessions before error. Saving what we have...")

    print(f"Download complete! Total {site} sessions retrieved: {len(sessions)}")
    
    # Save in the exact format expected by event_loader.py
    output_data = {
        "_meta": {
            "site": site,
            "description": "Full dataset downloaded via acnportal API"
        },
        "_items": sessions
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, cls=DateTimeEncoder)
    
    print(f"Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Download full ACN-Data dataset.")
    parser.add_argument("--token", type=str, required=True, help="Your ACN-Data API token.")
    parser.add_argument("--site", type=str, choices=["caltech", "jpl", "all"], default="all",
                        help="Which site to download (caltech, jpl, or all).")
    args = parser.parse_args()

    sites_to_download = ["caltech", "jpl"] if args.site == "all" else [args.site]
    
    base_dir = Path(__file__).resolve().parent / "charging"
    
    for site in sites_to_download:
        output_file = base_dir / f"{site}_sessions_full.json"
        download_site_data(args.token, site, output_file)


if __name__ == "__main__":
    main()
