class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        # sliding window
        l = 0
        curWin = set()
        longest = 0

        for r in range(len(s)):
            # Shrink
            while s[r] in curWin:
                curWin.remove(s[l])
                l += 1

            # Exapnd
            curWin.add(s[r])
            
            # Record
            longest = max(longest, len(curWin))
        
        return longest