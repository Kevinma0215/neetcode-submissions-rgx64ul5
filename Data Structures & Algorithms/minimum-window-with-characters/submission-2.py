class Solution:
    def minWindow(self, s: str, t: str) -> str:
        # Two string
        l = 0
        shortest = float("infinity")
        start = 0

        # store t in dict
        countT = {}
        for c in t:
            countT[c] = 1 + countT.get(c, 0)
        
        countWindow = {}
        
        have, need = 0, len(countT)

        for r in range(len(s)):
            c = s[r]
            # expanding
            countWindow[c] = 1 + countWindow.get(c, 0)

            if c in countT and countWindow[c] == countT[c]:
                have += 1

            # shrinking
            # while l < r:
            #     if s[l] not in countT:
            #         countWindow[s[l]] -= 1
            #         l += 1
            #         continue
            #     else:
            #         if countWindow[s[l]] > countT[s[l]]:
            #             countWindow[s[l]] -= 1
            #             l += 1
            #             continue
            #     break
            while l < r and (s[l] not in countT or countWindow[s[l]] > countT[s[l]]):
                countWindow[s[l]] -= 1
                l += 1

            # recording
            if have == need and (r - l + 1) < shortest:
                shortest = r - l + 1
                start = l

        return s[start: start + shortest] if shortest != float("infinity") else ""
