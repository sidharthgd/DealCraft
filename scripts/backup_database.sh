#!/bin/bash

# DealCraft Database Backup Script
# Run this before any major deployments or changes

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

PROJECT_ID=$(gcloud config get-value project)
INSTANCE_NAME="dealcraft-db"
DATABASE_NAME="dealcraft"
BACKUP_PREFIX="dealcraft-backup"

if [ -z "$PROJECT_ID" ]; then
    echo -e "${RED}Error: No GCP project configured${NC}"
    exit 1
fi

echo -e "${GREEN}🔄 Creating database backup...${NC}"

# Generate timestamp for backup name
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="${BACKUP_PREFIX}-${TIMESTAMP}"

# Create the backup
gcloud sql export sql $INSTANCE_NAME gs://${PROJECT_ID}_cloudbuild/${BACKUP_NAME}.sql \
    --database=$DATABASE_NAME \
    --offload

echo -e "${GREEN}✅ Backup created: ${BACKUP_NAME}.sql${NC}"
echo -e "${YELLOW}📍 Location: gs://${PROJECT_ID}_cloudbuild/${BACKUP_NAME}.sql${NC}"

# List recent backups
echo -e "${GREEN}📄 Recent backups:${NC}"
gsutil ls gs://${PROJECT_ID}_cloudbuild/${BACKUP_PREFIX}-* | tail -5

echo -e "${YELLOW}💡 To restore: gcloud sql import sql $INSTANCE_NAME gs://${PROJECT_ID}_cloudbuild/${BACKUP_NAME}.sql --database=$DATABASE_NAME${NC}"
