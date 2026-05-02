from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from mcp import ClientSession
from mcp.client.stdio import stdio_client
import asyncio

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

async def run():
    async with stdio_client(
        command=["python", "mcp_server.py"]
    ) as (read, write):

        async with ClientSession(read, write) as session:
            tools = await session.list_tools()
            print("Available MCP tools:", tools)

            # Ask the LLM a question
            user_prompt = "What are the top 5 customers by total spend?"

            # Manually prompt the model to generate SQL
            prompt = f"""
You have access to a database tool called `query_database`.
Return back the number of values the user requests

Question: {user_prompt}
"""

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(**inputs, max_new_tokens=256)
            sql_query = tokenizer.decode(outputs[0], skip_special_tokens=True)

            print("Received values:", sql_query)

            # Call MCP tool
            result = await session.call_tool(
                "query_database",
                arguments={"sql": sql_query}
            )

            print("Database Result:")
            print(result.content)

asyncio.run(run())

