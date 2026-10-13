class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        # Sliding window
        charSet = set()
        l = 0
        res = 0

        # if not repeated, r + 1 and update longest
        for r in range(len(s)):
            # Shrink
            while s[r] in charSet:
                charSet.remove(s[l])
                l += 1
            # Expand
            charSet.add(s[r])
            # Record
            res = max(res, r - l + 1)
        
        return res