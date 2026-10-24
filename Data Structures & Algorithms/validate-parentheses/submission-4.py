class Solution:
    def isValid(self, s: str) -> bool:
        pairs = {"(": ")",
                 "{": "}",
                 "[": "]" }
        
        rec = []

        for c in s:
            if c in pairs:
                rec.append(c)
            else:
                if not rec: return False
                t = rec.pop()
                if pairs[t] != c: return False

        return not rec