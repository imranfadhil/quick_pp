#!/bin/bash

# Script to run the database initialization with pgvector

echo "Starting PostgreSQL with pgvector and running database initialization..."

# Create necessary directories
mkdir -p data/postgres
mkdir -p data/langflow

# Run the services
docker-compose up --build -d

echo "Database initialization completed!"
echo "PostgreSQL with pgvector is running on port 5432"
echo "You can now start other services that depend on the database" 