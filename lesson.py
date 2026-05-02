class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        dummy: ListNode = ListNode(0)
        current: ListNode = dummy
        carry: int = 0

        while l1 or l2 or carry:
            val1: int = l1.val if l1 else 0
            val2: int = l2.val if l2 else 0

            total: int = val1 + val2 + carry

            digit: int = total % 10
            carry = total // 10

            current.next = ListNode(digit)
            

            print(f"val1: {val1}, val2: {val2}, total: {total}, digit: {digit}")

            break

        # while l1 or l2 or carry:
        #     val1: int = l1.val if l1 else 0
        #     val2: int = l2.val if l2 else 0

        #     total: int = val1 + val2 + carry

        #     digit = total % 10
        #     carry = total // 10
        #     current.next = ListNode(digit)
        #     current = current.next

        #     l1 = l1.next if l1 else None
        #     l2 = l2.next if l2 else None

        # return dummy.next

solution = Solution()
solution.addTwoNumbers(ListNode(2, ListNode(4, ListNode(3))), ListNode(5, ListNode(6, ListNode(4))))