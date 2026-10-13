class Solution:
    def minWindow(self, s: str, t: str) -> str:
        # Sliding window
        # Check
        if t == "":
            return ""

        # Store target in dict{char: freq}
        countT, window = {}, {}
        for c in t:
            countT[c] = 1 + countT.get(c, 0)
        
        # Check condition statement
        have, need = 0, len(countT)
        res, resLen = [-1, -1], float("infinity")

        l = 0
        for r in range(len(s)):
            c = s[r]
            window[c] = 1 + window.get(c, 0)

            if c in countT and window[c] == countT[c]:
                have += 1
            
            # When found a pair
            while have == need:
                # Check the shortest
                if (r - l + 1) < resLen:
                    res = [l, r]
                    resLen = r - l + 1

                # Shrink
                window[s[l]] -= 1
                if s[l] in countT and window[s[l]] < countT[s[l]]:
                    have -= 1
                l += 1
        
        l, r = res
        return s[l: r + 1] if resLen != float("infinity") else ""