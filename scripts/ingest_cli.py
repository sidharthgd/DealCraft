#!/usr/bin/env python3
"""
CLI tool for ingesting documents into DealCraft AI

Usage:
    python ingest_cli.py --deal "Deal Name" /path/to/documents/
    python ingest_cli.py --deal-id 123 /path/to/documents/
"""

import asyncio
import argparse
import sys
import os
from pathlib import Path
from typing import List

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent / "backend"))

from app.services.ingest import IngestService
from app.models import Deal, Document
from app.database import AsyncSessionLocal
from sqlalchemy import select

async def create_deal(name: str, description: str = None) -> int:
    """Create a new deal and return its ID"""
    async with AsyncSessionLocal() as db:
        # TODO: Get user_id from authentication or config
        user_id = 1  # Default user for CLI
        
        deal = Deal(
            name=name,
            description=description,
            user_id=user_id
        )
        
        db.add(deal)
        await db.commit()
        await db.refresh(deal)
        
        print(f"Created deal '{name}' with ID: {deal.id}")
        return deal.id

async def get_deal_id(deal_name: str) -> int:
    """Get deal ID by name"""
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(Deal).where(Deal.name == deal_name)
        )
        deal = result.scalar_one_or_none()
        
        if not deal:
            raise ValueError(f"Deal '{deal_name}' not found")
        
        return deal.id

async def ingest_file(file_path: Path, deal_id: int, ingest_service: IngestService) -> bool:
    """Ingest a single file"""
    try:
        # Create document record
        async with AsyncSessionLocal() as db:
            doc = Document(
                filename=file_path.name,
                file_path=str(file_path),
                file_size=file_path.stat().st_size,
                mime_type=get_mime_type(file_path),
                deal_id=deal_id,
                processing_status="pending"
            )
            
            db.add(doc)
            await db.commit()
            await db.refresh(doc)
            
            print(f"Processing: {file_path.name}")
            
            # Process the document
            success = await ingest_service.process_document(doc.id, str(file_path))
            
            if success:
                print(f"✅ Successfully processed: {file_path.name}")
            else:
                print(f"❌ Failed to process: {file_path.name}")
            
            return success
            
    except Exception as e:
        print(f"❌ Error processing {file_path.name}: {e}")
        return False

def get_mime_type(file_path: Path) -> str:
    """Get MIME type based on file extension"""
    extension = file_path.suffix.lower()
    mime_types = {
        '.pdf': 'application/pdf',
        '.txt': 'text/plain',
        '.doc': 'application/msword',
        '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        '.csv': 'text/csv',
        '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        '.xls': 'application/vnd.ms-excel'
    }
    return mime_types.get(extension, 'application/octet-stream')

def get_supported_files(directory: Path) -> List[Path]:
    """Get all supported files in directory"""
    supported_extensions = {'.pdf', '.txt', '.doc', '.docx', '.csv', '.xlsx', '.xls'}
    files = []
    
    for file_path in directory.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            files.append(file_path)
    
    return sorted(files)

async def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(description='Ingest documents into DealCraft AI')
    
    # Deal identification
    deal_group = parser.add_mutually_exclusive_group(required=True)
    deal_group.add_argument('--deal', type=str, help='Deal name (will be created if not exists)')
    deal_group.add_argument('--deal-id', type=int, help='Existing deal ID')
    
    # Options
    parser.add_argument('--description', type=str, help='Deal description (when creating new deal)')
    parser.add_argument('--recursive', '-r', action='store_true', help='Process directories recursively')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    # Path
    parser.add_argument('path', help='Path to file or directory to ingest')
    
    args = parser.parse_args()
    
    # Validate path
    path = Path(args.path)
    if not path.exists():
        print(f"❌ Path does not exist: {path}")
        sys.exit(1)
    
    try:
        # Initialize ingest service
        print("🔧 Initializing services...")
        ingest_service = IngestService()
        
        # Get or create deal
        if args.deal_id:
            deal_id = args.deal_id
            print(f"📁 Using existing deal ID: {deal_id}")
        else:
            try:
                deal_id = await get_deal_id(args.deal)
                print(f"📁 Found existing deal '{args.deal}' with ID: {deal_id}")
            except ValueError:
                deal_id = await create_deal(args.deal, args.description)
        
        # Get files to process
        if path.is_file():
            files = [path]
        else:
            print(f"📂 Scanning directory: {path}")
            files = get_supported_files(path)
        
        if not files:
            print("❌ No supported files found")
            sys.exit(1)
        
        print(f"📄 Found {len(files)} file(s) to process")
        
        # Process files
        successful = 0
        failed = 0
        
        for file_path in files:
            success = await ingest_file(file_path, deal_id, ingest_service)
            if success:
                successful += 1
            else:
                failed += 1
        
        # Summary
        print(f"\n📊 Processing complete:")
        print(f"   ✅ Successful: {successful}")
        print(f"   ❌ Failed: {failed}")
        print(f"   📁 Deal ID: {deal_id}")
        
        if failed > 0:
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 