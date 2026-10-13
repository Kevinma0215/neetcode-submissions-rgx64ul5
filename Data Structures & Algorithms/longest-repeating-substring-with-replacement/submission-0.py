class Solution:
    def characterReplacement(self, s: str, k: int) -> int:
        # Sliding Window
        longest = 0
        charSet = set(s)

        for c in charSet:
            l = 0
            count = 0
            for r in range(len(s)):
                # expand
                if s[r] == c:
                    count += 1
                else:
                    # shrink
                    while (r - l + 1) - count > k:
                        if s[l] == c:
                            count -= 1
                        l += 1                        

                # record
                longest = max(longest, (r - l + 1))
        
        return longest