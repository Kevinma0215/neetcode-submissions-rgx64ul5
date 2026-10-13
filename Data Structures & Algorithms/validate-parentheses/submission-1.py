class Solution:
    def isValid(self, s: str) -> bool:
        # Stack
        stack = []
        closeToOpen = { ")": "(", "]": "[", "}": "{"}

        for c in s:
            if c in closeToOpen:
                if stack and stack[-1] == closeToOpen[c]:
                    stack.pop() # list use pop()
                else:
                    return False
            else:
                stack.append(c)
        
        return not stack
