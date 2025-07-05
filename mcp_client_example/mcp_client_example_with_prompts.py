import asyncio
import json
import os
import shlex
import traceback
from contextlib import AsyncExitStack

import nest_asyncio
from anthropic import Anthropic
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Load configuration and environment
with open('../../config.json') as f:
    config = json.load(f)
os.environ["ANTHROPIC_API_KEY"] = config['ANTHROPIC_API_KEY']

nest_asyncio.apply()
load_dotenv()


class MCP_ChatBot:
    """
    MCP (Model Context Protocol) ChatBot that connects to multiple MCP servers
    and provides a conversational interface using Anthropic's Claude.
    
    Supports tools, prompts, and resources from connected MCP servers.
    """
    
    def __init__(self):
        self.exit_stack = AsyncExitStack()
        self.anthropic = Anthropic()
        self.available_tools = []      # Tools from all connected servers
        self.available_prompts = []    # Prompts from all connected servers  
        self.sessions = {}             # Maps tool/prompt/resource names to MCP sessions

    async def _register_server_capabilities(self, session, server_name):
        """Register tools, prompts and resources from a single server."""
        capabilities = [
            ("tools", session.list_tools, self._register_tools),
            ("prompts", session.list_prompts, self._register_prompts), 
            ("resources", session.list_resources, self._register_resources)
        ]
        
        for capability_name, list_method, register_method in capabilities:
            try:
                response = await list_method()
                await register_method(response, session)
            except Exception as e:
                print(f"Server {server_name} doesn't support {capability_name}: {e}")
    
    async def _register_tools(self, response, session):
        """Register tools from server response."""
        for tool in response.tools:
            self.sessions[tool.name] = session
            self.available_tools.append({
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema
            })
    
    async def _register_prompts(self, response, session):
        """Register prompts from server response."""
        if response and response.prompts:
            for prompt in response.prompts:
                self.sessions[prompt.name] = session
                self.available_prompts.append({
                    "name": prompt.name,
                    "description": prompt.description,
                    "arguments": prompt.arguments
                })
    
    async def _register_resources(self, response, session):
        """Register resources from server response."""
        if response and response.resources:
            for resource in response.resources:
                resource_uri = str(resource.uri)
                self.sessions[resource_uri] = session

    async def connect_to_server(self, server_name, server_config):
        """Connect to a single MCP server and register its capabilities."""
        try:
            server_params = StdioServerParameters(**server_config)
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            read, write = stdio_transport
            session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            await session.initialize()
            await self._register_server_capabilities(session, server_name)
                
        except Exception as e:
            print(f"Error connecting to {server_name}: {e}")
            traceback.print_exc()

    async def connect_to_servers(self):
        """Load server configuration and connect to all configured MCP servers."""
        try:
            with open("server_config.json", "r") as file:
                data = json.load(file)
            
            servers = data.get("mcpServers", {})
            for server_name, server_config in servers.items():
                await self.connect_to_server(server_name, server_config)
        except Exception as e:
            print(f"Error loading server config: {e}")
            traceback.print_exc()
            raise
    
    async def _find_matching_prompt(self, query):
        """Find a matching prompt for the given query using LLM evaluation."""
        if not self.available_prompts:
            return None
        
        # Use LLM to evaluate prompt relevance
        prompt_scores = []
        
        for prompt in self.available_prompts:
            # Create evaluation prompt for the LLM
            evaluation_prompt = f"""
You are an expert at evaluating whether a prompt template is relevant for a user query.

User Query: "{query}"

Prompt Template:
- Name: {prompt['name']}
- Description: {prompt['description']}

Rate the relevance of this prompt template for the user query on a scale of 0-5:
- 0: Completely irrelevant
- 1: Slightly relevant
- 2: Somewhat relevant  
- 3: Moderately relevant
- 4: Highly relevant
- 5: Perfect match

Consider:
- Does the prompt template address the user's intent?
- Would using this prompt template provide a better response than a generic query?
- Are the topics and context aligned?

Respond with only a single number (0-5) and no other text.
"""
            
            try:
                response = self.anthropic.messages.create(
                    max_tokens=10,
                    model='claude-3-5-sonnet-20241022',
                    messages=[{'role': 'user', 'content': evaluation_prompt}]
                )
                
                # Extract the score from the response
                score_text = response.content[0].text.strip()
                score = int(score_text)
                
                if score >= 3:  # Only consider prompts with score >= 3
                    prompt_scores.append((prompt, score))
                    
            except Exception as e:
                print(f"Error evaluating prompt {prompt['name']}: {e}")
                continue
        
        # Return the prompt with the highest score
        if prompt_scores:
            best_prompt, best_score = max(prompt_scores, key=lambda x: x[1])
            return best_prompt
        
        return None
    
    async def _combine_prompt_with_query(self, prompt_name, user_query):
        """Use LLM to combine prompt template with user query."""
        # First, get the prompt template content
        session = self.sessions.get(prompt_name)
        if not session:
            print(f"Prompt '{prompt_name}' not found.")
            return None
        
        try:
            # Find the prompt definition to get its arguments
            prompt_def = None
            for prompt in self.available_prompts:
                if prompt['name'] == prompt_name:
                    prompt_def = prompt
                    break
            
            # Prepare arguments for the prompt template
            args = {}
            if prompt_def and prompt_def.get('arguments'):
                for arg in prompt_def['arguments']:
                    arg_name = arg.name if hasattr(arg, 'name') else arg.get('name', '')
                    if arg_name:
                        # Use placeholder format for arguments
                        args[arg_name] = '<' + str(arg_name) + '>'
            
            # Get the prompt template with arguments
            result = await session.get_prompt(prompt_name, arguments=args)
            if not result or not result.messages:
                print(f"Could not retrieve prompt template for '{prompt_name}'")
                return None
            
            prompt_content = result.messages[0].content
            prompt_text = self._extract_prompt_text(prompt_content)
            
            # Create combination prompt for the LLM
            combination_prompt = f"""
You are an expert at combining prompt templates with user queries to create optimized prompts.

Original User Query: "{user_query}"

Prompt Template:
{prompt_text}

Your task:
1. Analyze the user's query and the prompt template
2. Combine them intelligently to create a single, coherent prompt
3. Ensure the user's specific question/request is addressed within the context of the template
4. Maintain the structure and intent of the template while incorporating the user's query

Respond with only the combined prompt text, no explanations or additional text.
"""
            
            response = self.anthropic.messages.create(
                max_tokens=2048,
                model='claude-3-5-sonnet-20241022',
                messages=[{'role': 'user', 'content': combination_prompt}]
            )
            
            return response.content[0].text.strip()
            
        except Exception as e:
            print(f"Error combining prompt with query: {e}")
            return None
    
    async def process_query(self, query):
        """Process a user query through Anthropic's Claude, handling tool calls iteratively."""
        # Check if there's a matching prompt first
        matching_prompt = await self._find_matching_prompt(query)
        
        if matching_prompt:
            print(f"Found matching prompt: {matching_prompt['name']}")
            print(f"Description: {matching_prompt['description']}")
            
            # Ask user if they want to use the prompt template
            use_prompt = input("Would you like to use this prompt template? (y/n): ").strip().lower()
            
            if use_prompt == 'y' or use_prompt == 'yes':
                print("Combining prompt template with your query...")
                
                # Use LLM to combine prompt template with user query
                combined_prompt = await self._combine_prompt_with_query(matching_prompt['name'], query)
                
                if combined_prompt:
                    print(f"Combined prompt created. Processing...")
                    # Process the combined prompt instead of the original query
                    messages = [{'role': 'user', 'content': combined_prompt}]
                else:
                    print("Failed to combine prompt template. Using original query.")
                    messages = [{'role': 'user', 'content': query}]
            else:
                # Use original query if user doesn't want to use the prompt
                messages = [{'role': 'user', 'content': query}]
        else:
            # Process the original query if no matching prompt found
            messages = [{'role': 'user', 'content': query}]

        # print(messages)
        
        # Process the final query (either original or combined)
        while True:
            response = self.anthropic.messages.create(
                max_tokens=2024,
                model='claude-3-5-sonnet-20241022', 
                tools=self.available_tools,
                messages=messages
            )
            
            assistant_content = []
            has_tool_use = False
            
            for content in response.content:
                if content.type == 'text':
                    print(content.text)
                    assistant_content.append(content)
                elif content.type == 'tool_use':
                    has_tool_use = True
                    assistant_content.append(content)
                    messages.append({'role': 'assistant', 'content': assistant_content})
                    
                    # Log tool call information
                    print(f"\n[TOOL CALL] Tool: {content.name}")
                    print(f"[TOOL CALL] Arguments: {json.dumps(content.input, indent=2)}")
                    
                    # Execute the tool call
                    session = self.sessions.get(content.name)
                    if not session:
                        print(f"Tool '{content.name}' not found.")
                        break
                        
                    result = await session.call_tool(content.name, arguments=content.input)
                    
                    # Log tool result
                    print(f"[TOOL RESULT] Tool: {content.name}")
                    print(f"[TOOL RESULT] Content: {result.content}")
                    
                    messages.append({
                        "role": "user", 
                        "content": [{
                            "type": "tool_result",
                            "tool_use_id": content.id,
                            "content": result.content
                        }]
                    })
            
            if not has_tool_use:
                break

    async def get_resource(self, resource_uri):
        """Retrieve and display content from an MCP resource."""
        session = self.sessions.get(resource_uri)
        
        # Fallback: find any session that handles this resource type
        if not session and resource_uri.startswith("changelog://"):
            session = next(
                (sess for uri, sess in self.sessions.items() 
                 if uri.startswith("changelog://")), 
                None
            )
            
        if not session:
            print(f"Resource '{resource_uri}' not found.")
            return
        
        try:
            result = await session.read_resource(uri=resource_uri)
            if result and result.contents:
                print(f"\nResource: {resource_uri}")
                print("Content:")
                print(result.contents[0].text)
            else:
                print("No content available.")
        except Exception as e:
            print(f"Error reading resource: {e}")
            traceback.print_exc()
    
    async def list_tools(self):
        """List all available tools."""
        if not self.available_tools:
            print("No tools available.")
            return
        
        print("\nAvailable tools:")
        for tool in self.available_tools:
            print(f"- {tool['name']}: {tool['description']}")
    
    async def list_prompts(self):
        """List all available prompts."""
        if not self.available_prompts:
            print("No prompts available.")
            return
        
        print("\nAvailable prompts:")
        for prompt in self.available_prompts:
            print(f"- {prompt['name']}: {prompt['description']}")
            if prompt['arguments']:
                print(f"  Arguments:")
                for arg in prompt['arguments']:
                    arg_name = arg.name if hasattr(arg, 'name') else arg.get('name', '')
                    print(f"    - {arg_name}")
    
    async def execute_tool(self, tool_name, args):
        """Execute an MCP tool directly with given arguments."""
        session = self.sessions.get(tool_name)
        if not session:
            print(f"Tool '{tool_name}' not found.")
            return
        
        try:
            result = await session.call_tool(tool_name, arguments=args)
            print(f"\nTool '{tool_name}' result:")
            print(result.content)
        except Exception as e:
            print(f"Error executing tool: {e}")
            traceback.print_exc()
    
    def _extract_prompt_text(self, prompt_content):
        """Extract text content from various prompt content formats."""
        if isinstance(prompt_content, str):
            return prompt_content
        elif hasattr(prompt_content, 'text'):
            return prompt_content.text
        else:
            # Handle list of content items
            return " ".join(
                item.text if hasattr(item, 'text') else str(item) 
                for item in prompt_content
            )

    async def execute_prompt(self, prompt_name, args):
        """Execute an MCP prompt with given arguments and process the result."""
        session = self.sessions.get(prompt_name)
        if not session:
            print(f"Prompt '{prompt_name}' not found.")
            return
        
        try:
            result = await session.get_prompt(prompt_name, arguments=args)
            if result and result.messages:
                prompt_content = result.messages[0].content
                text = self._extract_prompt_text(prompt_content)
                
                print(f"\nExecuting prompt '{prompt_name}'...")
                await self.process_query(text)
        except Exception as e:
            print(f"Error executing prompt: {e}")
            traceback.print_exc()
    
    def _parse_command_arguments(self, query):
        """Parse command line with proper handling of quoted strings."""
        try:
            return shlex.split(query)
        except ValueError:
            print("Error parsing command. Check your quotes.")
            return None
    
    def _clean_quoted_value(self, value):
        """Remove surrounding quotes from argument values."""
        if (value.startswith('"') and value.endswith('"')) or \
           (value.startswith("'") and value.endswith("'")):
            return value[1:-1]
        return value
    
    def _parse_prompt_arguments(self, args_list):
        """Parse key=value arguments for prompt execution."""
        args = {}
        for arg in args_list:
            if '=' in arg:
                key, value = arg.split('=', 1)
                args[key] = self._clean_quoted_value(value)
        return args

    async def chat_loop(self):
        """Main interactive chat loop with command processing."""
        print("\nMCP Chatbot Started!")
        print("Commands:")
        print("  quit                           - Exit the chatbot")
        print("  @periods                       - Show available changelog periods") 
        print("  @<period>                      - View changelog for specific period")
        print("  /tools                         - List available tools")
        print("  /tool <name> <arg1=value1>     - Execute a tool with arguments")
        print("  /prompts                       - List available prompts")
        print("  /prompt <name> <arg1=value1>   - Execute a prompt with arguments")
        
        while True:
            try:
                query = input("\nQuery: ").strip()
                if not query:
                    continue
        
                if query.lower() == 'quit':
                    break
                
                # Handle resource requests (@command)
                if query.startswith('@'):
                    period = query[1:]
                    resource_uri = "changelog://periods" if period == "periods" else f"changelog://{period}"
                    await self.get_resource(resource_uri)
                    continue
                
                # Handle slash commands
                if query.startswith('/'):
                    parts = self._parse_command_arguments(query)
                    if not parts:
                        continue
                        
                    command = parts[0].lower()
                    
                    if command == '/tools':
                        await self.list_tools()
                    elif command == '/tool':
                        if len(parts) < 2:
                            print("Usage: /tool <name> <arg1=value1> <arg2=value2>")
                            continue
                        
                        tool_name = parts[1]
                        args = self._parse_prompt_arguments(parts[2:])
                        await self.execute_tool(tool_name, args)
                    elif command == '/prompts':
                        await self.list_prompts()
                    elif command == '/prompt':
                        if len(parts) < 2:
                            print("Usage: /prompt <name> <arg1=value1> <arg2=value2>")
                            continue
                        
                        prompt_name = parts[1]
                        args = self._parse_prompt_arguments(parts[2:])
                        await self.execute_prompt(prompt_name, args)
                    else:
                        print(f"Unknown command: {command}")
                    continue
                
                # Process regular queries
                await self.process_query(query)
                    
            except Exception as e:
                print(f"\nError in chat loop: {e}")
                traceback.print_exc()

    async def cleanup(self):
        """Clean up resources and close all connections."""
        await self.exit_stack.aclose()


async def main():
    """Main entry point for the MCP ChatBot application."""
    chatbot = MCP_ChatBot()
    try:
        await chatbot.connect_to_servers()
        await chatbot.chat_loop()
    finally:
        await chatbot.cleanup()


if __name__ == "__main__":
    asyncio.run(main())