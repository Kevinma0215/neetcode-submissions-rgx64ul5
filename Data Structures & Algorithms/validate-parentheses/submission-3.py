class Solution:
    def isValid(self, s: str) -> bool:
        # Stack
        pair = {"(": ")",
                "{": "}",
                "[": "]"}
        
        stack = []

        for c in s:
            if c in pair:
                stack.append(c)
            else:
                if not stack or pair[stack.pop()] != c: 
                    return False

        return True if stack == [] else False
               
