class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        mp = {}
        l = 0
        res = 0

        for r in range(len(s)):
            # Shrink
            if s[r] in mp:
                l = max(mp[s[r]] + 1, l)
            # Expand
            mp[s[r]] = r
            # Record
            res = max(res, r - l + 1)

        return res