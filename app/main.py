"""
Main FastAPI application entry point.
"""
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Depends, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.openapi.utils import get_openapi
from fastapi.security import HTTPBearer

from app.api.endpoints import router as api_router
from app.core.config import settings
from app.core.logging import logger
from app.api.deps import get_current_user, User


# Security scheme
security_scheme = HTTPBearer()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan events for FastAPI app.
    Initialize resources at startup and clean up at shutdown.
    """
    logger.info("Starting AI Task Copilot application")
    # Initialize agent, clients, and connections here
    # e.g., initialize Weaviate client, Supabase client, etc.
    
    # Initialize Supabase realtime subscription (if needed)
    # This would be the place to initialize any global realtime subscriptions
    
    yield
    # Clean up resources here
    logger.info("Shutting down AI Task Copilot application")


# Create FastAPI app
app = FastAPI(
    title="AI Task Copilot",
    description="Agentic assistant for Notion, Slack, and GitHub using LLMs",
    version="0.1.0",
    lifespan=lifespan,
)

# Custom OpenAPI to use HTTP Bearer security scheme
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    
    # Add HTTPBearer security scheme
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT"
        }
    }
    
    # Add security to all routes except auth login/register and WebSockets
    for path in openapi_schema["paths"]:
        # Skip login, register, and WebSocket routes
        if (not path.endswith("/login") and 
            not path.endswith("/register") and 
            not path.endswith("/login-page") and 
            not "websocket" in path.lower() and
            path != "/health" and 
            path != "/"):
            
            for method in openapi_schema["paths"][path]:
                openapi_schema["paths"][path][method]["security"] = [{"BearerAuth": []}]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please try again later."},
    )


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Welcome to AI Task Copilot API", "status": "healthy"}


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


# Example protected endpoint
@app.get("/protected-resource", response_class=HTMLResponse)
async def protected_resource(request: Request, current_user: User = Depends(get_current_user)):
    """Example of a protected endpoint that requires authentication."""
    # Create a user-friendly representation of the authenticated user
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Protected Resource</title>
        <style>
            body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
            h1 {{ color: #4CAF50; }}
            pre {{ background-color: #f5f5f5; padding: 10px; border-radius: 5px; }}
            .success {{ color: #4CAF50; }}
            .token-section {{ margin: 20px 0; }}
            button {{ padding: 8px 15px; background-color: #4CAF50; color: white; border: none; 
                     border-radius: 4px; cursor: pointer; margin-right: 10px; }}
            button:hover {{ background-color: #45a049; }}
        </style>
    </head>
    <body>
        <h1>Authentication Successful! 🔐</h1>
        <p class="success">You have successfully accessed a protected resource. This page requires authentication.</p>
        
        <h2>User Information:</h2>
        <ul>
            <li><strong>User ID:</strong> {current_user.id}</li>
            <li><strong>Email:</strong> {current_user.email}</li>
        </ul>
        
        <div class="token-section">
            <h2>Your Authentication Token:</h2>
            <div id="token-display">Loading token...</div>
        </div>
        
        <div>
            <button onclick="window.location.href='/api/auth/login-page'">Back to Login Page</button>
            <button onclick="logoutUser()">Logout</button>
        </div>
        
        <script>
            // Display the token from localStorage
            document.addEventListener('DOMContentLoaded', function() {{
                try {{
                    const token = localStorage.getItem('token');
                    const tokenDisplay = document.getElementById('token-display');
                    
                    if (token) {{
                        // Display the token (first 20 chars)
                        const shortToken = token.substring(0, 20) + '...';
                        tokenDisplay.innerHTML = '<pre>' + shortToken + '</pre>';
                        tokenDisplay.innerHTML += '<p>Token is securely stored in your browser.</p>';
                    }} else {{
                        tokenDisplay.innerHTML = '<p>No token found in localStorage. Authentication may have bypassed client storage.</p>';
                    }}
                }} catch (error) {{
                    console.error("Error displaying token:", error);
                }}
            }});
            
            // Logout function
            function logoutUser() {{
                try {{
                    // Clear token from localStorage
                    localStorage.removeItem('token');
                    
                    // Call logout endpoint
                    fetch('/api/auth/logout', {{
                        method: 'POST',
                        headers: {{
                            'Authorization': 'Bearer ' + localStorage.getItem('token')
                        }}
                    }}).then(response => {{
                        // Redirect to login page
                        window.location.href = '/api/auth/login-page';
                    }}).catch(error => {{
                        console.error("Logout error:", error);
                        // Still redirect even if the API call fails
                        window.location.href = '/api/auth/login-page';
                    }});
                }} catch (error) {{
                    console.error("Logout error:", error);
                    window.location.href = '/api/auth/login-page';
                }}
            }}
        </script>
    </body>
    </html>
    """


# Dashboard endpoint to show realtime task status
@app.get("/tasks-dashboard", response_class=HTMLResponse)
async def tasks_dashboard(request: Request, current_user: User = Depends(get_current_user)):
    """
    Simple dashboard to view real-time task status.
    This provides a basic UI for testing the WebSocket functionality.
    """
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Task Status Dashboard</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }
            h1 { color: #4169E1; }
            .tasks-container { margin-top: 20px; }
            .task-card { 
                border: 1px solid #ccc; 
                margin-bottom: 10px; 
                padding: 15px; 
                border-radius: 5px;
                transition: background-color 0.3s;
            }
            .task-header { display: flex; justify-content: space-between; }
            .task-title { font-weight: bold; margin-bottom: 5px; }
            .task-status {
                padding: 3px 8px;
                border-radius: 10px;
                font-size: 12px;
                font-weight: bold;
            }
            .status-pending { background-color: #FFF9C4; color: #FF6F00; }
            .status-in_progress { background-color: #E3F2FD; color: #1565C0; }
            .status-completed { background-color: #E8F5E9; color: #2E7D32; }
            .status-failed { background-color: #FFEBEE; color: #C62828; }
            .task-result { 
                margin-top: 10px; 
                background-color: #f5f5f5; 
                padding: 10px; 
                border-radius: 5px; 
                white-space: pre-wrap;
                overflow-wrap: break-word;
            }
            .task-meta { 
                font-size: 12px; 
                color: #666; 
                margin-top: 5px;
                display: flex;
                justify-content: space-between;
            }
            .no-tasks { 
                padding: 20px; 
                background-color: #f5f5f5; 
                border-radius: 5px; 
                text-align: center;
                color: #666;
            }
            .connection-status {
                padding: 5px 10px;
                margin-bottom: 10px;
                border-radius: 5px;
                font-size: 14px;
            }
            .connected { background-color: #E8F5E9; color: #2E7D32; }
            .disconnected { background-color: #FFEBEE; color: #C62828; }
            .debug-info {
                margin: 10px 0;
                padding: 10px;
                background-color: #f5f5f5;
                border-radius: 5px;
                font-size: 12px;
            }
            button {
                padding: 8px 15px;
                background-color: #4169E1;
                color: white;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                margin-right: 10px;
            }
            button:hover { background-color: #3D5AFE; }
            .highlight { animation: highlight 2s; }
            @keyframes highlight {
                0% { background-color: #FFF9C4; }
                100% { background-color: transparent; }
            }
        </style>
    </head>
    <body>
        <h1>Real-time Task Status Dashboard</h1>
        
        <div id="connection-status" class="connection-status disconnected">
            Connecting to WebSocket...
        </div>
        
        <div class="debug-info">
            <p><strong>Auth Info:</strong></p>
            <p>User ID: """ + current_user.id + """</p>
            <p>Email: """ + current_user.email + """</p>
            <p>Token in localStorage: <span id="token-info">Checking...</span></p>
        </div>
        
        <div>
            <button onclick="refreshTasks()">Refresh Tasks</button>
            <button onclick="window.location.href='/api/tasks/run-task'">Create New Task</button>
            <button onclick="window.location.href='/debug-auth'">Debug Auth</button>
        </div>
        
        <div id="tasks-container" class="tasks-container">
            <div class="no-tasks">No tasks found. Create a new task or check your connection.</div>
        </div>
        
        <script>
            // Store the WebSocket connection
            let ws;
            let token = localStorage.getItem('token');
            
            // Display token info
            function displayTokenInfo() {
                const tokenInfo = document.getElementById('token-info');
                if (token) {
                    const tokenLen = token.length;
                    tokenInfo.textContent = `Yes (${tokenLen} chars, starts with: ${token.substring(0, 10)}...)`;
                } else {
                    tokenInfo.textContent = 'No token found!';
                }
            }
            
            // Connect to WebSocket
            function connectWebSocket() {
                const connectionStatus = document.getElementById('connection-status');
                connectionStatus.className = 'connection-status disconnected';
                connectionStatus.textContent = 'Connecting to WebSocket...';
                
                // Create WebSocket connection with token
                ws = new WebSocket(`ws://${window.location.host}/api/realtime/tasks-feed?token=${token}`);
                
                ws.onopen = function(event) {
                    connectionStatus.className = 'connection-status connected';
                    connectionStatus.textContent = 'Connected to WebSocket';
                    console.log('WebSocket connected');
                };
                
                ws.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    console.log('WebSocket message received:', data);
                    
                    if (data.type === 'tasks_list') {
                        renderTasksList(data.data);
                    } else if (data.type === 'task_update') {
                        updateTask(data.data);
                    }
                };
                
                ws.onclose = function(event) {
                    connectionStatus.className = 'connection-status disconnected';
                    connectionStatus.textContent = 'Disconnected from WebSocket';
                    console.log('WebSocket disconnected, reconnecting in 5 seconds...');
                    
                    // Attempt to reconnect after 5 seconds
                    setTimeout(connectWebSocket, 5000);
                };
                
                ws.onerror = function(error) {
                    console.error('WebSocket error:', error);
                    connectionStatus.className = 'connection-status disconnected';
                    connectionStatus.textContent = `WebSocket error: ${error.message || 'Unknown error'}`;
                };
            }
            
            // Render tasks list
            function renderTasksList(tasks) {
                const container = document.getElementById('tasks-container');
                
                if (!tasks || tasks.length === 0) {
                    container.innerHTML = '<div class="no-tasks">No tasks found. Create a new task or check your connection.</div>';
                    return;
                }
                
                container.innerHTML = '';
                
                // Sort tasks by created_at (newest first)
                tasks.sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
                
                tasks.forEach(task => {
                    container.appendChild(createTaskElement(task));
                });
            }
            
            // Create task element
            function createTaskElement(task) {
                const taskElement = document.createElement('div');
                taskElement.className = 'task-card';
                taskElement.id = `task-${task.task_id}`;
                
                const createdDate = task.created_at ? new Date(task.created_at).toLocaleString() : 'Unknown';
                const updatedDate = task.updated_at ? new Date(task.updated_at).toLocaleString() : 'Unknown';
                
                taskElement.innerHTML = `
                    <div class="task-header">
                        <div class="task-title">${task.description}</div>
                        <div class="task-status status-${task.status}">${task.status}</div>
                    </div>
                    ${task.result ? `<div class="task-result">${task.result}</div>` : ''}
                    <div class="task-meta">
                        <div>Created: ${createdDate}</div>
                        <div>Updated: ${updatedDate}</div>
                    </div>
                `;
                
                return taskElement;
            }
            
            // Update a single task
            function updateTask(task) {
                const container = document.getElementById('tasks-container');
                const existingTask = document.getElementById(`task-${task.task_id}`);
                
                if (existingTask) {
                    // Update existing task
                    const newTask = createTaskElement(task);
                    container.replaceChild(newTask, existingTask);
                    newTask.classList.add('highlight');
                } else {
                    // Add new task to the beginning
                    const newTask = createTaskElement(task);
                    container.insertBefore(newTask, container.firstChild);
                    newTask.classList.add('highlight');
                    
                    // Remove no-tasks message if present
                    const noTasks = container.querySelector('.no-tasks');
                    if (noTasks) {
                        container.removeChild(noTasks);
                    }
                }
            }
            
            // Refresh tasks
            function refreshTasks() {
                if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.send('refresh');
                } else {
                    connectWebSocket();
                }
            }
            
            // Initialize when the page loads
            document.addEventListener('DOMContentLoaded', function() {
                displayTokenInfo();
                
                if (!token) {
                    alert('No authentication token found in localStorage. Please log in.');
                    window.location.href = '/api/auth/login-page';
                    return;
                }
                
                connectWebSocket();
            });
        </script>
    </body>
    </html>
    """


# Add after the root endpoint
@app.get("/debug-auth", response_class=HTMLResponse)
async def debug_auth():
    """Debug endpoint to test authentication."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Authentication Debug</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            h1 { color: #4169E1; }
            textarea { width: 100%; height: 100px; }
            button { padding: 8px 15px; background-color: #4169E1; color: white; border: none; 
                     border-radius: 4px; cursor: pointer; margin-right: 10px; }
            .result { margin-top: 20px; padding: 10px; background-color: #f5f5f5; border-radius: 5px; }
            .error { color: red; }
        </style>
    </head>
    <body>
        <h1>Authentication Debug Tool</h1>
        
        <div>
            <h2>Current Token</h2>
            <div id="current-token">No token found in localStorage</div>
            <button onclick="refreshTokenDisplay()">Refresh</button>
            <button onclick="clearToken()">Clear Token</button>
        </div>
        
        <div style="margin-top: 20px;">
            <h2>Set New Token</h2>
            <textarea id="new-token" placeholder="Paste your JWT token here"></textarea>
            <button onclick="saveToken()">Save Token</button>
        </div>
        
        <div style="margin-top: 20px;">
            <h2>Test Protected Endpoint</h2>
            <input id="endpoint" value="/protected-resource" style="width: 300px;"/>
            <button onclick="testEndpoint()">Test</button>
        </div>
        
        <div id="test-result" class="result" style="display: none;"></div>
        
        <script>
            // Display current token from localStorage
            function refreshTokenDisplay() {
                const tokenDisplay = document.getElementById('current-token');
                const token = localStorage.getItem('token');
                
                if (token) {
                    // Display first and last 10 chars of token
                    const tokenLen = token.length;
                    const maskedToken = token.substring(0, 10) + '...' + token.substring(tokenLen - 10);
                    tokenDisplay.innerHTML = '<div><code>' + maskedToken + '</code></div>';
                    tokenDisplay.innerHTML += '<div>Token length: ' + tokenLen + ' characters</div>';
                } else {
                    tokenDisplay.textContent = 'No token found in localStorage';
                }
            }
            
            // Save new token to localStorage
            function saveToken() {
                const tokenInput = document.getElementById('new-token');
                const token = tokenInput.value.trim();
                
                if (token) {
                    localStorage.setItem('token', token);
                    tokenInput.value = '';
                    refreshTokenDisplay();
                    alert('Token saved to localStorage');
                } else {
                    alert('Please enter a token');
                }
            }
            
            // Clear token from localStorage
            function clearToken() {
                localStorage.removeItem('token');
                refreshTokenDisplay();
                alert('Token cleared from localStorage');
            }
            
            // Test a protected endpoint with the current token
            function testEndpoint() {
                const resultDiv = document.getElementById('test-result');
                resultDiv.style.display = 'block';
                resultDiv.innerHTML = 'Testing...';
                
                const token = localStorage.getItem('token');
                const endpoint = document.getElementById('endpoint').value;
                
                if (!token) {
                    resultDiv.innerHTML = '<div class="error">No token found in localStorage</div>';
                    return;
                }
                
                fetch(endpoint, {
                    headers: {
                        'Authorization': 'Bearer ' + token
                    }
                })
                .then(response => {
                    resultDiv.innerHTML = '<div>Status: ' + response.status + ' ' + response.statusText + '</div>';
                    
                    // Try to get response text
                    return response.text().then(text => {
                        try {
                            // Try to parse as JSON first
                            const json = JSON.parse(text);
                            resultDiv.innerHTML += '<div>Response:</div><pre>' + JSON.stringify(json, null, 2) + '</pre>';
                        } catch (e) {
                            // If not JSON, show as text
                            resultDiv.innerHTML += '<div>Response:</div><div>' + text.substring(0, 500) + 
                              (text.length > 500 ? '...' : '') + '</div>';
                        }
                    });
                })
                .catch(error => {
                    resultDiv.innerHTML = '<div class="error">Error: ' + error.message + '</div>';
                });
            }
            
            // Initialize token display
            document.addEventListener('DOMContentLoaded', refreshTokenDisplay);
        </script>
    </body>
    </html>
    """


# Include API routers
app.include_router(api_router, prefix=settings.API_PREFIX)


