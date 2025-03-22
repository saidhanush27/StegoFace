# Python3 code to demonstrate working of
# Converting String to binary
# Using join() + ord() + format()

# initializing string
test_str = "GeeksforGeeks"

# printing original string
print("The original string is : " + str(test_str))

# using join() + ord() + format()
# Converting String to binary
res = ''.join(format(ord(i), '08b') for i in test_str)

# printing result
print("The string after binary conversion : " + str(res))
# Binary data
b = str(res)  # binary for 'geek'

# Split the binary string into chunks of 8 bits (1 byte)
n = [b[i:i+8] for i in range(0, len(b), 8)]

# Convert binary to string
s = ''.join(chr(int(i, 2)) for i in n)

# Output the result
print(s)