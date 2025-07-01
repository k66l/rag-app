#!/bin/bash

# RAG Application Docker Helper Script
# This script provides convenient commands for Docker operations

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is installed and running
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        print_error "Docker is not running. Please start Docker first."
        exit 1
    fi
}

# Build the Docker image
build() {
    print_status "Building RAG Application Docker image..."
    docker build -t rag-app:latest .
    print_success "Build completed successfully!"
}

# Run the application in production mode
run_production() {
    print_status "Starting RAG Application in production mode..."
    docker run -d \
        --name rag-app \
        -p 8000:8000 \
        --memory="2g" \
        --cpus="1.0" \
        -v "$(pwd)/logs:/app/logs" \
        -v "$(pwd)/temp_docs:/app/temp_docs" \
        --env-file .env \
        --restart unless-stopped \
        rag-app:latest
    print_success "Application started! Access it at http://localhost:8000"
}

# Run the application in development mode
run_development() {
    print_status "Starting RAG Application in development mode..."
    docker run -it \
        --name rag-app-dev \
        -p 8000:8000 \
        -v "$(pwd)/app:/app/app" \
        -v "$(pwd)/tests:/app/tests" \
        -v "$(pwd)/logs:/app/logs" \
        --env-file .env \
        -e DEBUG=true \
        rag-app:latest \
        uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
}

# Use Docker Compose for full stack
compose_up() {
    print_status "Starting RAG Application with Docker Compose..."
    docker-compose up -d
    print_success "Application started! Access it at http://localhost:8000"
}

# Use Docker Compose for development
compose_dev() {
    print_status "Starting RAG Application in development mode with Docker Compose..."
    docker-compose --profile dev up
}

# Stop all containers
stop() {
    print_status "Stopping RAG Application containers..."
    docker stop rag-app rag-app-dev 2>/dev/null || true
    docker-compose down 2>/dev/null || true
    print_success "All containers stopped!"
}

# Remove containers and images
clean() {
    print_status "Cleaning up Docker resources..."
    
    # Stop containers
    docker stop rag-app rag-app-dev 2>/dev/null || true
    
    # Remove containers
    docker rm rag-app rag-app-dev 2>/dev/null || true
    
    # Remove images
    docker rmi rag-app:latest 2>/dev/null || true
    
    # Clean up compose resources
    docker-compose down --volumes --remove-orphans 2>/dev/null || true
    
    # Clean up unused resources
    docker system prune -f
    
    print_success "Cleanup completed!"
}

# Show logs
logs() {
    container_name=${1:-rag-app}
    print_status "Showing logs for container: $container_name"
    if docker ps --format "table {{.Names}}" | grep -q "^$container_name$"; then
        docker logs -f "$container_name"
    else
        print_error "Container '$container_name' is not running"
        exit 1
    fi
}

# Show container status
status() {
    print_status "Docker container status:"
    echo
    docker ps --filter "name=rag-app" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    echo
    
    if docker ps --filter "name=rag-app" | grep -q rag-app; then
        print_success "RAG Application is running!"
        echo "🌐 Web Interface: http://localhost:8000"
        echo "📚 API Documentation: http://localhost:8000/docs"
        echo "❤️  Health Check: http://localhost:8000/"
    else
        print_warning "RAG Application is not running"
    fi
}

# Run tests inside container
test() {
    print_status "Running tests in Docker container..."
    docker run --rm \
        -v "$(pwd)/tests:/app/tests" \
        rag-app:latest \
        python run_tests.py --type all --verbose
}

# Execute command inside container
exec_cmd() {
    container_name=${1:-rag-app}
    shift
    command=${*:-bash}
    
    print_status "Executing command in container: $container_name"
    if docker ps --format "table {{.Names}}" | grep -q "^$container_name$"; then
        docker exec -it "$container_name" $command
    else
        print_error "Container '$container_name' is not running"
        exit 1
    fi
}

# Show help
show_help() {
    echo "RAG Application Docker Helper Script"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  build           Build the Docker image"
    echo "  run             Run in production mode"
    echo "  dev             Run in development mode"
    echo "  compose         Start with Docker Compose (production)"
    echo "  compose-dev     Start with Docker Compose (development)"
    echo "  stop            Stop all containers"
    echo "  clean           Remove containers and clean up"
    echo "  logs [name]     Show logs (default: rag-app)"
    echo "  status          Show container status"
    echo "  test            Run tests in container"
    echo "  exec [name] [cmd] Execute command in container"
    echo "  help            Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 build                    # Build the image"
    echo "  $0 run                      # Start in production mode"
    echo "  $0 dev                      # Start in development mode"
    echo "  $0 logs                     # Show application logs"
    echo "  $0 exec rag-app bash        # Open bash in container"
    echo "  $0 test                     # Run test suite"
    echo ""
}

# Main script logic
main() {
    # Check if Docker is available
    check_docker
    
    # Handle commands
    case "${1:-help}" in
        build)
            build
            ;;
        run|production)
            build
            run_production
            ;;
        dev|development)
            build
            run_development
            ;;
        compose|up)
            compose_up
            ;;
        compose-dev|dev-compose)
            compose_dev
            ;;
        stop|down)
            stop
            ;;
        clean|cleanup)
            clean
            ;;
        logs)
            logs "$2"
            ;;
        status|ps)
            status
            ;;
        test|tests)
            build
            test
            ;;
        exec|execute)
            exec_cmd "$2" "${@:3}"
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            print_error "Unknown command: $1"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@" 