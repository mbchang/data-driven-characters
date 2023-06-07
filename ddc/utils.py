import math


def apply_file_naming_convention(text):
    """Apply file naming conventions to a string."""
    text = text.replace("(", '"').replace(")", '"')
    return text.replace('"', "-").replace(" ", "_")


def order_of_magnitude(number):
    """Return the order of magnitude of a number."""
    if number == 0:
        return 0
    else:
        return math.floor(math.log10(abs(number)))
