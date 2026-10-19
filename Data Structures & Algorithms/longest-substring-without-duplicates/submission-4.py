class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        # Sliding window
        window = set()
        longest = 0

        l = 0
        for r in range(len(s)):
            # shrink
            while s[r] in window:
                window.remove(s[l])
                l += 1

            # explore
            window.add(s[r])        

            # record
            longest = max(longest, len(window))

        return longest