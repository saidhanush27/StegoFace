# Function to convert a string to its binary representation
def string_to_binary(input_string):
    """
    Converts a string to its binary equivalent.
    Each character is converted to an 8-bit binary representation.
    """
    return ''.join(format(ord(char), '08b') for char in input_string)


# Function to convert binary back to string
def binary_to_string(binary_string):
    """
    Converts a binary string back to its original string.
    Assumes the binary string is a valid sequence of 8-bit ASCII characters.
    """
    n = 8
    return ''.join(chr(int(binary_string[i:i + n], 2)) for i in range(0, len(binary_string), n))


# Function to calculate the parity bits for Hamming(7,4) encoding
def calculate_parity_bits(data_bits):
    """
    Calculate the parity bits for a 4-bit data using Hamming(7,4) encoding.
    """
    if len(data_bits) < 7:
        binary_list = [0] * (7 - len(data_bits)) + data_bits  # Pad with zeros at the beginning

    # Continue with your parity bit calculations...
    # Assuming `binary_list` has at least 7 elements now.
    # Here we use dummy calculations for illustration:
    binary_list[0] = (binary_list[2] + binary_list[4] + binary_list[6]) % 2  # Example P1 calculation
    binary_list[1] = (binary_list[2] + binary_list[5] + binary_list[6]) % 2  # Example P2 calculation
    binary_list[2] = (binary_list[4] + binary_list[5] + binary_list[6]) % 2  # Example P3 calculation

    return binary_list





# Function to detect and correct errors in a received codeword using Hamming(7,4)
def detect_and_correct_error(codeword):
    """
    Detect and correct single-bit errors in a Hamming(7,4) codeword.
    Returns the corrected codeword and the position of the error (if any).
    """
    # Convert codeword to list of integers
    code = [int(bit) for bit in codeword]

    # Calculate the parity bits
    p1 = (code[2] + code[4] + code[6]) % 2
    p2 = (code[2] + code[5] + code[6]) % 2
    p3 = (code[4] + code[5] + code[6]) % 2

    # Check parity
    parity_bits = [p1, p2, p3]

    # Calculate error position
    error_position = p1 * 1 + p2 * 2 + p3 * 4

    if error_position != 0:
        # Flip the bit at the error position (1-based index)
        print(f"Error detected at position: {error_position}")
        code[error_position - 1] = 1 - code[error_position - 1]  # Flip the bit
        print(f"Corrected codeword: {''.join(str(bit) for bit in code)}")
        return ''.join(str(bit) for bit in code)
    else:
        print("No error detected")
        return codeword


# Example Usage
# Step 1: Convert the string to binary
input_string = "becc"  # Example string
binary_string = string_to_binary(input_string)
print(f"Original String: {input_string}")
print(f"Binary Representation: {binary_string}")

# Step 2: Apply Hamming(7,4) encoding for error correction
encoded_data = calculate_parity_bits([int(bit) for bit in binary_string[:4]])  # Encode the first 4 bits of the binary string
print(f"Encoded data with parity bits: {encoded_data}")

# Step 3: Simulate an error by flipping a random bit (for demonstration)
error_encoded_data = encoded_data[:]
error_encoded_data = error_encoded_data[:4] + [0] + error_encoded_data[5:] # Simulate an error at bit position 5
print(f"Encoded data with error: {error_encoded_data}")

# Step 4: Detect and correct errors in the received codeword
corrected_data = detect_and_correct_error(error_encoded_data)

# Step 5: Convert the corrected binary back to the string
# We first need to remove the parity bits before converting back to string (extracting only the original data)
corrected_data_without_parity = corrected_data[2] + corrected_data[4] + corrected_data[5] + corrected_data[6]
decoded_string = binary_to_string(corrected_data_without_parity)
print(f"Decoded String after Error Correction: {decoded_string}")
