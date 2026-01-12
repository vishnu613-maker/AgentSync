#!/usr/bin/env python3
"""
Zapier MCP Tool Discovery Script (FINAL FIX)
Handles Server-Sent Events (SSE) response format from Zapier MCP
"""

import json
import httpx
import sys

# Your Zapier MCP Webhook URL
ZAPIER_WEBHOOK = "https://mcp.zapier.com/api/mcp/s/YWUzZGUyZDAtMGQzNy00OThkLWIxMWUtNWU4Y2M3ODg4ZGQ0OjdkM2I2YjdmLTY4MjAtNGE1Mi1iOTFmLWNjYWNhNGZiODVmNw==/mcp"

def parse_sse_response(response_text):
    """Parse Server-Sent Events (SSE) format response"""
    lines = response_text.strip().split('\n')
    result = None
    
    for line in lines:
        if line.startswith('data: '):
            data_str = line[6:]  # Remove 'data: ' prefix
            try:
                result = json.loads(data_str)
            except json.JSONDecodeError:
                pass
    
    return result


def discover_tools():
    """Discover all tools available in the Zapier MCP webhook"""
    
    print("🔍 Discovering Zapier MCP Tools...")
    print(f"📡 Webhook: {ZAPIER_WEBHOOK}\n")
    
    # Prepare request
    payload = {
        "jsonrpc": "2.0",
        "id": "1",
        "method": "tools/list",
        "params": {}
    }
    
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream"
    }
    
    try:
        response = httpx.post(
            ZAPIER_WEBHOOK,
            json=payload,
            headers=headers,
            timeout=30
        )
        
        print(f"📊 Response Status: {response.status_code}\n")
        
        if response.status_code == 200:
            # ✅ Parse SSE response format
            print("📝 Raw Response (for debugging):")
            print(response.text[:500])
            print("...\n")
            
            result = parse_sse_response(response.text)
            
            if not result:
                print("❌ Could not parse response!")
                print(f"Full response:\n{response.text}")
                return
            
            # Check for errors
            if "error" in result:
                print(f"❌ Error: {result['error']}")
                return
            
            # Extract tools
            tools = result.get("result", {}).get("tools", [])
            
            if not tools:
                print("⚠️  No tools found!")
                return
            
            print(f"✅ Found {len(tools)} Tools:\n")
            print("=" * 100)
            
            # Display each tool with details
            for i, tool in enumerate(tools, 1):
                print(f"\n[{i}] {tool.get('name', 'Unknown')}")
                print(f"    Description: {tool.get('description', 'No description')}")
                
                # Get input schema if available
                input_schema = tool.get("inputSchema", {})
                properties = input_schema.get("properties", {})
                required = input_schema.get("required", [])
                
                if properties:
                    print(f"    Parameters:")
                    for param_name, param_info in properties.items():
                        required_str = "✓ Required" if param_name in required else "○ Optional"
                        param_type = param_info.get("type", "string")
                        param_desc = param_info.get("description", "")
                        print(f"      • {param_name} ({param_type}) - {required_str}")
                        if param_desc:
                            print(f"        └─ {param_desc}")
                else:
                    print(f"    Parameters: None")
            
            print("\n" + "=" * 100)
            
            # Generate summary for code
            print("\n📋 Tool Names for Your Code (tasks_agent.py):\n")
            print("self.tools = [")
            for tool in tools:
                name = tool.get("name", "")
                desc = tool.get("description", "")
                input_schema = tool.get("inputSchema", {})
                required = input_schema.get("required", [])
                optional = [
                    p for p in input_schema.get("properties", {}).keys()
                    if p not in required
                ]
                
                internal_name = name.lower().replace(" ", "_")
                
                print(f'    {{')
                print(f'        "name": "{internal_name}",')
                print(f'        "zapier_name": "{name}",')
                print(f'        "description": "{desc}",')
                print(f'        "required_params": {json.dumps(required)},')
                print(f'        "optional_params": {json.dumps(optional)},')
                print(f'        "intent_keywords": []')
                print(f'    }},')
            print("]\n")
            
            # Export tools.json
            print("\n💾 Saved to 'zapier_tools.json':\n")
            tools_data = {
                "tools": [
                    {
                        "name": tool.get("name", "").lower().replace(" ", "_"),
                        "zapier_name": tool.get("name", ""),
                        "description": tool.get("description", ""),
                        "required_params": tool.get("inputSchema", {}).get("required", []),
                        "optional_params": [
                            p for p in tool.get("inputSchema", {}).get("properties", {}).keys()
                            if p not in tool.get("inputSchema", {}).get("required", [])
                        ]
                    }
                    for tool in tools
                ]
            }
            
            # Save to file
            with open("zapier_tools.json", "w") as f:
                json.dump(tools_data, f, indent=2)
            
            print(json.dumps(tools_data, indent=2))
            
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except httpx.TimeoutException:
        print("❌ Timeout! The webhook took too long to respond.")
    except Exception as e:
        print(f"❌ Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


def test_tool(tool_name, parameters):
    """Test calling a specific tool"""
    
    print(f"\n🧪 Testing Tool: {tool_name}")
    print(f"📝 Parameters: {json.dumps(parameters, indent=2)}\n")
    
    payload = {
        "jsonrpc": "2.0",
        "id": "2",
        "method": tool_name,
        "params": parameters
    }
    
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream"
    }
    
    try:
        response = httpx.post(
            ZAPIER_WEBHOOK,
            json=payload,
            headers=headers,
            timeout=30
        )
        
        print(f"📊 Response Status: {response.status_code}\n")
        
        if response.status_code == 200:
            result = parse_sse_response(response.text)
            print("✅ Response:")
            print(json.dumps(result, indent=2))
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {type(e).__name__}: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "list":
            discover_tools()
        elif sys.argv[1] == "test" and len(sys.argv) > 2:
            tool_name = sys.argv[2]
            parameters = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
            test_tool(tool_name, parameters)
        else:
            print("Usage:")
            print("  python3 discover_zapier_tools_FINAL.py list")
            print("  python3 discover_zapier_tools_FINAL.py test 'Create Task' '{\"task_list\": \"Work\", \"title\": \"test\"}'")
    else:
        discover_tools()
