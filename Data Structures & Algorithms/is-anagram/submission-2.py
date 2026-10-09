class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        if len(s) != len(t):
            return False

        counterS, counterT = {}, {} # declare once

        for i in range(len(s)):
            # hash map {} use .get(key, [other val]) to call the value
            counterS[s[i]] = 1 + counterS.get(s[i], 0)  
            counterT[t[i]] = 1 + counterT.get(t[i], 0)
        
        return counterS == counterT # hash {} can directly compare using ==