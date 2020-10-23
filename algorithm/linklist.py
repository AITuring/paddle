class Node(object):
    def __init__(self,item):
        self.item = item
        self.next = None

class LinkList(object):
    def __init__(self):
        self._head = None
    def is_empty(self):
        """判断链表是否为空"""
        return self._head is None
    def length(self):
        """计算链表长度"""
        cur = self._head
        count = 0
        while cur is not None:
            count+=1
            cur = cur.next
        return count
    def items(self):
        """遍历链表"""
        cur = self._head
        while cur is not None:
            yield cur.item
            cur = cur.next