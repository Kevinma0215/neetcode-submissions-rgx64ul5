class Solution:

    def encode(self, strs: List[str]) -> str:
        res = []
        for s in strs:
            res.append(str(len(s)))
            res.append("#") # why can use #?
            res.append(s)
        
        return "".join(res) # why need ""?

    def decode(self, s: str) -> List[str]:
        res = []
        i = 0

        while i < len(s):
            j = i
            while s[j] != '#':
                j += 1
            length = int(s[i:j])    # note [start: till]
            i = j + 1 
            j = i + length

            res.append(s[i:j])
            i = j
        
        return res

            
