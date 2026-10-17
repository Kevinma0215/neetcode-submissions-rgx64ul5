class Solution:
    def isValid(self, s: str) -> bool:
        # Stack
        matcher = {"(": ")",
                   "[": "]",
                   "{": "}"}

        checker = []
        val = False

        for c in s:
            # append
            if c in matcher:
                checker.append(c)
            # dissolve
            else:
                if not checker: return False
                if matcher[checker.pop()] != c: return False
        
        if not checker: 
            val = True

        return val
        
