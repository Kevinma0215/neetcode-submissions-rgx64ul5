class Solution:
    def minWindow(self, s: str, t: str) -> str:
        # Sliding window
        l = 0

        countT, countS = {}, {}
        for c in t:
            countT[c] = 1 + countT.get(c, 0)

        have, need = 0, len(countT)

        shortest = float("infinity")
        res = ""
        start = 0

        for r in range(len(s)):
            # explore
            if s[r] in countT:
                countS[s[r]] = 1 + countS.get(s[r], 0)
            
                if countS[s[r]] == countT[s[r]]:
                    have += 1

            # shrink
            if have == need:
                while s[l] not in countT or countS[s[l]] > countT[s[l]]:
                    if s[l] in countS: countS[s[l]] -= 1
                    l += 1

                # record
                if (r - l + 1) < shortest:
                    shortest = r - l + 1
                    res = s[l: r + 1]

        return res 
