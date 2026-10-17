class Solution:
    def minWindow(self, s: str, t: str) -> str:
        # Sliding window
        l = 0
        countT, countWin = {}, {}
        for c in t:
            countT[c] = 1 + countT.get(c, 0)

        have, need = 0, len(countT)

        shortest = float("infinity")
        start = 0
        res = ""

        for r in range(len(s)):
            if s[r] in countT:
                # Expand
                countWin[s[r]] = 1 + countWin.get(s[r], 0)
            
                # check have
                if countT[s[r]] == countWin.get(s[r], 0):
                    have += 1
            
            if have == need:
                # Shrink
                while l < r:
                    if s[l] not in countT:
                        l += 1
                        continue
                    else:
                        if countWin[s[l]] > countT[s[l]]:
                            countWin[s[l]] -= 1
                            l += 1
                            continue
                    break

                # Record
                if shortest > r - l + 1:
                    shortest = r - l + 1
                    start = l                
                

        return s[start: start + shortest] if shortest != float("infinity") else ""