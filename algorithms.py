from PIL import Image


def encode_text_to_image(text, image_path, output_image_path):
    # Convert the text to binary
    binary_text = ''.join(format(ord(char), '08b') for char in text)

    # Open the image
    img = Image.open(image_path)
    img = img.convert('RGB')  # Ensure it's in RGB format
    pixels = img.load()

    # Ensure the image is large enough to store the text
    width, height = img.size
    if len(binary_text) > width * height * 3:  # Each pixel can hold 3 bytes (RGB)
        raise ValueError("The image is too small to hold the given text")

    # Encode the binary text into the image
    idx = 0
    for y in range(height):
        for x in range(width):
            if idx < len(binary_text):
                pixel = list(pixels[x, y])  # Get the RGB values of the pixel
                for i in range(3):  # Modify the R, G, B values
                    if idx < len(binary_text):
                        pixel[i] = pixel[i] & ~1 | int(binary_text[idx])  # Set LSB to binary data
                        idx += 1
                pixels[x, y] = tuple(pixel)  # Put the modified pixel back into the image

    # Save the image with the encoded text
    img.save(output_image_path)
    print(f"Text successfully encoded into image and saved as {output_image_path}")


# Example Usage:
encode_text_to_image("100001011,", "static/uploads/aadhar_1.jpg", "encoded_image.png")
def decode_text_from_image(image_path):
    # Open the encoded image
    img = Image.open(image_path)
    img = img.convert('RGB')
    pixels = img.load()

    # Extract the binary data from the image
    binary_text = ''
    width, height = img.size
    for y in range(height):
        for x in range(width):
            pixel = pixels[x, y]
            for i in range(3):
                binary_text += str(pixel[i] & 1)  # Extract the LSB from each color channel

    # Convert the binary data back to text
    decoded_text = ''
    for i in range(0, len(binary_text), 8):
        byte = binary_text[i:i + 8]
        decoded_text += chr(int(byte, 2))
        if decoded_text[-1] == '\x00':  # Stop when we encounter the null character
            break

    print(f"Decoded Text: {decoded_text}")
    return decoded_text

# Example Usage:
decoded_text = decode_text_from_image("encoded_image.png")
txt = decoded_text

x = txt.split(',')

print(x)


def is_binary_string(s):
    # use set comprehension to extract all unique characters from the string
    unique_chars = {c for c in s}
    # check if the unique characters are only 0 and 1
    return unique_chars.issubset({'0', '1'})


def is_binary_string(s):
    # use set comprehension to extract all unique characters from the string
    unique_chars = {c for c in s}
    # check if the unique characters are only 0 and 1
    return unique_chars.issubset({'0', '1'})


s = is_binary_string(x[0])
print(s)
if s == "True":
    b = str(decoded_text)  # binary for 'geek'