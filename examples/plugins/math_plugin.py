"""
MathPlugin - Demonstrates native functions for mathematical operations.

This plugin shows how to create custom Python functions that AI can call.
"""

from semantic_kernel.functions import kernel_function
from typing import Annotated


class MathPlugin:
    """A plugin that provides mathematical operations."""
    
    @kernel_function(
        name="add",
        description="Adds two numbers together and returns the result."
    )
    def add(
        self,
        number1: Annotated[float, "The first number to add"],
        number2: Annotated[float, "The second number to add"],
    ) -> Annotated[float, "The sum of the two numbers"]:
        """Add two numbers."""
        result = number1 + number2
        print(f"[MathPlugin.add] {number1} + {number2} = {result}")
        return result
    
    @kernel_function(
        name="subtract",
        description="Subtracts the second number from the first number."
    )
    def subtract(
        self,
        number1: Annotated[float, "The number to subtract from"],
        number2: Annotated[float, "The number to subtract"],
    ) -> Annotated[float, "The difference"]:
        """Subtract two numbers."""
        result = number1 - number2
        print(f"[MathPlugin.subtract] {number1} - {number2} = {result}")
        return result
    
    @kernel_function(
        name="multiply",
        description="Multiplies two numbers together."
    )
    def multiply(
        self,
        number1: Annotated[float, "The first number"],
        number2: Annotated[float, "The second number"],
    ) -> Annotated[float, "The product"]:
        """Multiply two numbers."""
        result = number1 * number2
        print(f"[MathPlugin.multiply] {number1} × {number2} = {result}")
        return result
    
    @kernel_function(
        name="divide",
        description="Divides the first number by the second number."
    )
    def divide(
        self,
        number1: Annotated[float, "The dividend"],
        number2: Annotated[float, "The divisor"],
    ) -> Annotated[float, "The quotient"]:
        """Divide two numbers."""
        if number2 == 0:
            return "Error: Cannot divide by zero"
        result = number1 / number2
        print(f"[MathPlugin.divide] {number1} ÷ {number2} = {result}")
        return result
    
    @kernel_function(
        name="power",
        description="Raises a number to a power (base^exponent)."
    )
    def power(
        self,
        base: Annotated[float, "The base number"],
        exponent: Annotated[float, "The exponent"],
    ) -> Annotated[float, "The result of base raised to exponent"]:
        """Calculate power."""
        result = base ** exponent
        print(f"[MathPlugin.power] {base}^{exponent} = {result}")
        return result
    
    @kernel_function(
        name="square_root",
        description="Calculates the square root of a number."
    )
    def square_root(
        self,
        number: Annotated[float, "The number to find square root of"],
    ) -> Annotated[float, "The square root"]:
        """Calculate square root."""
        if number < 0:
            return "Error: Cannot calculate square root of negative number"
        result = number ** 0.5
        print(f"[MathPlugin.square_root] √{number} = {result}")
        return result
    
    @kernel_function(
        name="percentage",
        description="Calculates what percentage one number is of another."
    )
    def percentage(
        self,
        part: Annotated[float, "The part value"],
        whole: Annotated[float, "The whole value"],
    ) -> Annotated[float, "The percentage"]:
        """Calculate percentage."""
        if whole == 0:
            return "Error: Cannot calculate percentage with zero as whole"
        result = (part / whole) * 100
        print(f"[MathPlugin.percentage] {part} is {result}% of {whole}")
        return result

