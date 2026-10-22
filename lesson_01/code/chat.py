# Cretaes chat for insurance company
# Implements dialog with AI
import sys

from common.llm import get_client, estimate_cost, warn_if_truncated

MODEL = "claude-haiku-5-5"  # cheapest model for learinng
MAX_TOKENS = 1000  # max response length
SYSTEM = """You are a support assistant of "Shomer" insurance company.
- Answer in the user's language (Hebrew, Russian or English).
- Be brief: 2-4 sentences.
- You have no access to policy data. If asked about a specific policy,
  ask for the policy number and say a human agent will follow up.
- Never invent prices, coverage terms or deadlines."""

# define history
history = []


#################
def create_bot_response(client, user_input):
    # append user's input to the history
    history.append({"role": "user", "content": user_input})

    print("bot> ", end="")

    # create output stream
    with client.messages.stream(model=MODEL, max_tokens=MAX_TOKENS, system=SYSTEM, output_config={"effort": "low"},messages=history) as stream:
        for chunk in stream.text_stream:       # text deltas only, thinking is skipped
            print(chunk, end="", flush=True)
        reply = stream.get_final_message()     # full Message: content, usage, stop_reason

    # end the streamed line, move to the new line
    print()

    history.append({"role": "assistant", "content": reply.content})
    warn_if_truncated(reply)

    return reply


def main() -> int:

    # create client
    client = get_client()

    # condition to break the loop
    exit_cmd = ("/exit")

    # condition to reset
    reset_cmd = ("/reset")

    # define total costs value
    total_cost = 0.0

    print("Shomer support. Commands: /reset, /exit")

    while True:
        # Get input from the user
        user_input = input('you>').strip()

        # Check if the user wants to leave
        if user_input.lower() == exit_cmd:
            print("bot> Goodbye!")
            break  # This breaks out of the while loop

        # Check if the user wants to reset history
        if user_input.lower() == reset_cmd:
            history.clear()
            print("bot> the history was reset")
            continue

        # Process the input if the user doesn't type anything
        if not user_input:
            print("bot> You didn't say anything. Use /exit to exit or /reset to reset history")
            continue  # Skips the rest of the loop and prompts again
            
        # Generate the bot's response
        reply = create_bot_response(client, user_input)

        # calculate estimate costs by model
        turn_cost = estimate_cost(reply.usage, MODEL)
        total_cost += turn_cost
        print(f"     [in {reply.usage.input_tokens} | out {reply.usage.output_tokens} "
              f"| turn ${turn_cost:.6f} | total ${total_cost:.6f}]")

        

if __name__ == "__main__":
    sys.exit(main())