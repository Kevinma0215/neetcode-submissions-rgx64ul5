# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

class Solution:
    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        # BFS
        queueP = deque([p])
        queueQ = deque([q])

        while queueP and queueQ:
            for _ in range(len(queueP)):
                nodeP = queueP.popleft()
                nodeQ = queueQ.popleft()

                if nodeP is None and nodeQ is None:
                    continue

                if nodeP is None or nodeQ is None or nodeP.val != nodeQ.val: 
                    return False

                queueP.append(nodeP.left)
                queueP.append(nodeP.right)
                queueQ.append(nodeQ.left)
                queueQ.append(nodeQ.right)
        
        return True

