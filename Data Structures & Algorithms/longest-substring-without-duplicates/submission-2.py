class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        # sliding window
        l = 0
        longest = 0
        cur = set()

        for r in range(len(s)):
            c = s[r]

            # shrinking
            while c in cur:
                cur.remove(s[l])
                l += 1

            # expanding
            cur.add(c)

            # recording
            longest = max(longest, len(cur))

        return longest