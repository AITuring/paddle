def monotoneIncreasingDigits(N: int) -> int:
    ones = 111111111
    result = 0
    for _ in range(9):
        print(ones,result,_)
        while result + ones > N:
            ones //= 10
        result += ones
    return result

N=332
print(monotoneIncreasingDigits(N))