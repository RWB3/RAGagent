# tools/calculator.py
import logging

def run(expression):
    """Evaluates a mathematical expression."""
    try:
        result = eval(expression)  # WARNING: Be extremely careful with eval()!
        return str(result)
    except Exception as e:
        logging.error(f"Error evaluating expression: {e}")
        return "Error: Invalid expression."

if __name__ == '__main__':
    # Example usage (for testing)
    print(run("2 + 2"))
    print(run("10 / 0"))