package algorist

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"sort"
	"testing"
)

// Chapter 2: Algorithm Analysis.

func TestSorts(t *testing.T) {
	tt := [][2][]int{
		{{3, 2, 1}, {1, 2, 3}},
		{{1, 2, 3}, {1, 2, 3}},
		{{2, 1, 3}, {1, 2, 3}},
		{{}, {}},
		{nil, nil},
	}

	sorts := []struct {
		name string
		f    func([]int) []int
	}{
		{"selectionSort", selectionSort_ch_2_5_1},
		{"insertionSort", insertionSort_ch_2_5_2},
	}

	for _, tc := range tt {
		tc := tc
		for _, srt := range sorts {
			srt := srt
			t.Run(fmt.Sprintf("%s(%+v)->%+v", srt.name, tc[0], tc[1]), func(t *testing.T) {
				t.Parallel()

				got := srt.f(tc[0])
				if len(got) != len(tc[1]) {
					t.Fatalf("want %v, got %v", tc[1], got)
				}

				for i := range got {
					if tc[1][i] != got[i] {
						t.Fatalf("item want[%d] = %d, got[%d] = %d", i, tc[1][i], i, got[i])
					}
				}
			})
		}
	}
}

func selectionSort_ch_2_5_1(items []int) []int {
	if items == nil {
		return nil
	}

	n := len(items)
	s := make([]int, n)
	copy(s, items)

	for i := 0; i < n; i++ {
		min := i
		for j := i + 1; j < n; j++ {
			if s[j] >= s[min] {
				continue
			}
			min = j
		}

		s[i], s[min] = s[min], s[i]
	}

	return s
}

func insertionSort_ch_2_5_2(items []int) []int {
	if items == nil {
		return nil
	}

	n := len(items)
	s := make([]int, n)
	copy(s, items)

	for i := 1; i < n; i++ {
		for j := i; j > 0 && s[j] < s[j-1]; j-- {
			s[j-1], s[j] = s[j], s[j-1]
		}
	}

	return s
}

func TestStringPatternMatching(t *testing.T) {
	tt := []struct {
		p, t string
		m    int
	}{
		{"", "as", 0},
		{"", "", 0},
		{"abba", "abaababbaba", 5},
		{"abbad", "abaababbaba", -1},
	}

	for _, tc := range tt {
		tc := tc
		t.Run(fmt.Sprintf("findmatch(%s, %s)=%d", tc.p, tc.t, tc.m), func(t *testing.T) {
			if mgot := findmatch_ch_2_5_3(tc.p, tc.t); mgot != tc.m {
				t.Fatalf("want %d, got %d", tc.m, mgot)
			}
		})
	}
}

func findmatch_ch_2_5_3(p, t string) int {
	pn, tn := len(p), len(t)

	for i := 0; i <= tn-pn; i++ {
		if t[i:i+pn] != p {
			continue
		}
		return i
	}
	return -1
}

func TestMatrixmult(t *testing.T) {
	tt := []struct {
		a, b, c matrix
	}{
		{
			a: matrix{rows: 2, columns: 3, m: [][]int{[]int{2, 1, 3}, []int{3, 5, 2}}},
			b: matrix{rows: 3, columns: 2, m: [][]int{[]int{4, 1}, []int{1, 2}, []int{2, 4}}},
			c: matrix{rows: 2, columns: 2, m: [][]int{[]int{15, 16}, []int{21, 21}}},
		},
	}

	for _, tc := range tt {
		tc := tc

		t.Run("", func(t *testing.T) {
			c := matrixmult_ch_2_5_4(tc.a, tc.b)

			for i := range c.m {
				for j := range c.m[i] {
					if v1, v2 := tc.c.m[i][j], c.m[i][j]; v1 != v2 {
						t.Fatalf("not eq: tc.c.m[%d][%d] = %d, c.m[%d][%d] = %d", i, j, v1, i, j, v2)
					}
				}
			}
		})
	}
}

type matrix struct {
	rows, columns int
	m             [][]int
}

// no matrices equality tests since we assume they always hold.
func matrixmult_ch_2_5_4(a, b matrix) matrix {
	c := matrix{rows: a.rows, columns: b.columns, m: make([][]int, a.rows)}

	for i := 0; i < c.rows; i++ {
		c.m[i] = make([]int, b.columns)
		for j := 0; j < c.columns; j++ {

			for k := 0; k < b.rows; k++ {
				c.m[i][j] += a.m[i][k] * b.m[k][j]
			}
		}
	}

	return c
}

// Chapter 3: Data Structures.

// arr is a fixed size array.
var arr [3]int

var arrdyn []int

func TestArray(t *testing.T) { arrdyn = arr[:]; fmt.Println(arrdyn) } // to avoid unused linter error.

// Chapter 3.1.2

func TestList_3_1_1(t *testing.T) {
	eqT := func(l0, l1 *list_ch_3_1_2) {
		t.Helper()
		if !lists_eq_help_3_1_2(l0, l1) {
			t.Fatal("lists not eq")
		}
	}

	eqT(nil, searchList_ch_3_1_2(nil, 0))
	tlist := insertList_ch_3_1_2(nil, 0)
	reflist := &list_ch_3_1_2{item: 0}
	eqT(reflist, tlist)
	eqT(reflist, searchList_ch_3_1_2(tlist, 0))

	tlist = insertList_ch_3_1_2(tlist, 1)
	eqT(reflist, searchList_ch_3_1_2(tlist, 0))
	reflist = &list_ch_3_1_2{next: reflist, item: 1}
	eqT(reflist, searchList_ch_3_1_2(tlist, 1))

	eqT(nil, deleteList_ch_3_1_2(tlist, 2, predecessorList_recursive_ch_3_1_2))
	eqT(nil, deleteList_ch_3_1_2(tlist, 2, predecessorList_ch_3_1_2))

	eqT(&list_ch_3_1_2{item: 1}, deleteList_ch_3_1_2(tlist, 1, predecessorList_recursive_ch_3_1_2))
	tlist = insertList_ch_3_1_2(tlist, 1)
	eqT(&list_ch_3_1_2{item: 1}, deleteList_ch_3_1_2(tlist, 1, predecessorList_ch_3_1_2))
}

func lists_eq_help_3_1_2(l0, l1 *list_ch_3_1_2) bool {
	if l0 == nil && l1 == nil {
		return true
	} else if l0 == nil || l1 == nil || l0.item != l1.item {
		return false
	}
	return lists_eq_help_3_1_2(l0.next, l1.next)
}

type list_ch_3_1_2 struct {
	item int            // data item
	next *list_ch_3_1_2 // point to successor
}

func searchList_ch_3_1_2(l *list_ch_3_1_2, item int) *list_ch_3_1_2 {
	if l == nil {
		return nil
	}
	if l.item == item {
		return l
	}

	return searchList_ch_3_1_2(l.next, item)
}

// l is assumed is not nil, but can point to nil list.
func insertList_ch_3_1_2(l *list_ch_3_1_2, item int) *list_ch_3_1_2 {
	return &list_ch_3_1_2{item, l}
}

func deleteList_ch_3_1_2(l *list_ch_3_1_2, item int, predecessor func(*list_ch_3_1_2, int) *list_ch_3_1_2) *list_ch_3_1_2 {
	if l == nil {
		return nil
	}
	if l.item == item {
		l.next = nil
		return l
	}

	p := predecessor(l, item)
	if p == nil {
		return nil
	}
	d := p.next
	p.next = nil
	if n := d.next; n != nil {
		d.next = nil
		p.next = n
	}

	return d
}

func predecessorList_recursive_ch_3_1_2(l *list_ch_3_1_2, item int) *list_ch_3_1_2 {
	if l.next == nil {
		return nil
	} else if l.next.item == item {
		return l
	}
	return predecessorList_recursive_ch_3_1_2(l.next, item)
}

func predecessorList_ch_3_1_2(l *list_ch_3_1_2, item int) *list_ch_3_1_2 {
	for ; ; l = l.next {
		if l == nil || l.next == nil {
			return nil
		}
		if l.next.item == item {
			return l
		}
	}

	return nil
}

// Binary search tree 3.4.1
type bstitem struct {
	key string
	val int
}

type bstree struct {
	item                bstitem
	parent, left, right *bstree
}

func search_bst(t *bstree, k string) *bstree {
	if t == nil || t.item.key == k {
		return t
	}
	if t.item.key > k {
		return search_bst(t.left, k)
	}

	return search_bst(t.right, k)
}

func min_bst(t *bstree) *bstree {
	if t == nil {
		return nil
	} else if t.left == nil {
		return t
	}

	return min_bst(t.left)
}

func max_bst(t *bstree) *bstree {
	if t == nil {
		return nil
	} else if t.right == nil {
		return t
	}

	return max_bst(t.right)
}

func traverse_inorder_bst(t *bstree, f func(*bstree)) {
	if t != nil {
		traverse_inorder_bst(t.left, f)
		f(t)
		traverse_inorder_bst(t.right, f)
	}
}

func insert_bst(t **bstree, p *bstree, x bstitem) {
	if *t == nil {
		*t = &bstree{item: x, parent: p}
		return
	}

	if (*t).item.key <= x.key {
		insert_bst(&((*t).right), *t, x)
		return
	}

	insert_bst(&((*t).left), *t, x)
}

func del_bst(t **bstree, x string) {
	if *t == nil {
		return
	} else if (*t).item.key < x {
		del_bst(&((*t).right), x)
		return
	} else if (*t).item.key > x {
		del_bst(&((*t).left), x)
		return
	}

	// rm t
	if (*t).left == nil && (*t).right == nil {
		// no child
		*t = nil
		return
	} else if ((*t).left == nil) != ((*t).right == nil) {
		// one chlid
		n := (*t).left
		if n == nil {
			n = (*t).right
		}
		(*t).item = n.item
		(*t).left, (*t).right = nil, nil
		return
	}

	n := max_bst((*t).left) // find a node to replace t.
	if np := n.parent; np.item.key <= n.item.key {
		np.right = nil
	} else {
		np.left = nil
	}
	(*t).item = n.item
}

func TestInsertDeleteBST(t *testing.T) {
	var root, ref *bstree
	eqtest := func() {
		if !eq_bst(root, ref) {
			t.Fatalf("not eq, ref %#v, root %#v", ref, root)
		}
	}
	eqtest()

	insert_bst(&root, nil, bstitem{"5", 5})
	ref = &bstree{item: bstitem{"5", 5}}
	eqtest()

	insert_bst(&root, nil, bstitem{"1", 1})
	ref.left = &bstree{item: bstitem{"1", 1}, parent: ref}
	eqtest()

	insert_bst(&root, nil, bstitem{"8", 8})
	ref.right = &bstree{item: bstitem{"8", 8}, parent: ref}
	eqtest()

	insert_bst(&root, nil, bstitem{"7", 7})
	ref.right.left = &bstree{item: bstitem{"7", 7}, parent: ref.right}
	eqtest()

	insert_bst(&root, nil, bstitem{"6", 6})
	ref.right.left.left = &bstree{item: bstitem{"6", 6}, parent: ref.right.left}
	eqtest()

	insert_bst(&root, nil, bstitem{"2", 2})
	ref.left.right = &bstree{item: bstitem{"2", 2}, parent: ref.left}
	eqtest()

	del_bst(&root, "9")
	eqtest()

	var nilbst *bstree
	del_bst(&nilbst, "9")

	del_bst(&root, "5")
	ref.item = bstitem{"2", 2}
	ref.left.right = nil
	eqtest()

	del_bst(&root, "1")
	ref.left = nil
	eqtest()

	del_bst(&root, "7")
	ref.right.left.item = bstitem{"6", 6}
	ref.right.left.left = nil
	eqtest()

	// remove all keys
	keys := []string{}
	traverse_inorder_bst(root, func(b *bstree) {
		keys = append(keys, b.item.key)
	})
	for _, k := range keys {
		del_bst(&root, k)
	}
	ref = nil
	eqtest()
}

func eq_bst(t0, t1 *bstree) bool {
	if t0 == nil && t1 == nil {
		return true
	}

	if t0.item != t1.item {
		return false
	}

	return eq_bst(t0.left, t1.left) && eq_bst(t0.right, t1.right)
}

// 3.10 Excersizes

// A common problem for compilers and text editors is determining whether
// the parentheses in a string are balanced and properly nested. For example, the
// string ((())())() contains properly nested pairs of parentheses, while the strings
// )()( and ()) do not. Give an algorithm that returns true if a string contains
// properly nested and balanced parentheses, and false if otherwise. For full credit,
// identify the position of the first offending parenthesis if the string is not properly
// nested and balanced.
//
// ((())())() -> true
// )()(       -> false
// ())        -> flase
func testBalancedPths_3_10_1(s string) bool {
	q := make([]rune, 0, len(s))
	open := map[rune]struct{}{'[': {}, '(': {}, '{': {}}
	clse := map[rune]rune{')': '(', ']': '[', '}': '{'}
	for _, r := range s {
		_, o := open[r]
		opener, c := clse[r]
		if !(o || c) {
			continue
		}
		if o {
			q = append(q, r) // push
			continue
		}

		n := len(q)
		if len(q) == 0 {
			return false
		}

		last := q[n-1]
		q = q[:n-1]
		if last != opener {
			return false
		}
	}

	return len(q) == 0
}

func TestBalancedParantheses_3_10_1(t *testing.T) {
	tt := []struct {
		s    string
		want bool
	}{
		{"((())())()", true},
		{")()(", false},
		{"())", false},
	}

	for _, tc := range tt {
		tc := tc
		t.Run(fmt.Sprintf("%s -> %v", tc.s, tc.want), func(t *testing.T) {
			got := testBalancedPths_3_10_1(tc.s)
			if tc.want != got {
				t.Fail()
			}
		})
	}
}

// [5] Give an algorithm that takes a string S consisting of opening and closing
// parentheses, say )()(())()()))())))(, and finds the length of the longest balanced
// parentheses in S, which is 12 in the example above. (Hint: The solution is not
// necessarily a contiguous run of parenthesis from S.)
//
// )()(())()()))())))( -> 12
func longestBalancesPrnthsLen_3_10_2(s string) int {
	plen := 0 // accumulates len, returned value
	q := make([]rune, 0, len(s))
	open := map[rune]struct{}{'[': {}, '(': {}, '{': {}}
	clse := map[rune]rune{')': '(', ']': '[', '}': '{'}
	for _, r := range s {
		_, o := open[r]
		if o {
			q = append(q, r) // push
			continue
		}

		n := len(q)
		if len(q) == 0 { // to not panic
			continue
		}

		opener := clse[r]
		last := q[n-1]
		q = q[:n-1]
		if last != opener {
			q = q[:0]
			continue
		}
		plen += 2
	}

	return plen
}

func TestLongesBalancedParantheses_3_10_2(t *testing.T) {
	tt := []struct {
		s    string
		want int
	}{
		{")()(())()()))())))(", 12},
	}

	for _, tc := range tt {
		tc := tc
		t.Run(fmt.Sprintf("%s -> %v", tc.s, tc.want), func(t *testing.T) {
			got := longestBalancesPrnthsLen_3_10_2(tc.s)
			if tc.want != got {
				t.Fail()
			}
		})
	}
}

// [3] Give an algorithm to reverse the direction of a given singly linked list. In
// other words, after the reversal all pointers should now point backwards. Your
// algorithm should take linear time.
//
// nil -> nil
// 3 - 2 - 1 - 0 -> 0 - 1 - 2 - 3
type litem_3_10_3 struct {
	item int
	next *litem_3_10_3
}

func reverseLinkedList_3_10_3(litem *litem_3_10_3) *litem_3_10_3 {
	if litem == nil {
		return nil
	}

	h, _ := reverseLinkedList_3_10_3_recurse(litem)
	return h
}

func reverseLinkedList_3_10_3_recurse(litem *litem_3_10_3) (*litem_3_10_3, *litem_3_10_3) {
	tail := &litem_3_10_3{litem.item, nil}
	if litem.next == nil {
		return tail, tail
	}

	head, oldtail := reverseLinkedList_3_10_3_recurse(litem.next)
	oldtail.next = tail
	return head, tail
}

func TestReverseLinkedList_3_10_3(t *testing.T) {
	list := buildLinkedListHelper_3_10_3([]int{1, 2, 3, 0})
	p, ints := 0, [4]int{}
	walkLitemsHelper_3_10_3_recurse(list, func(l *litem_3_10_3) {
		ints[p] = l.item
		p++
	})
	if want := [4]int{1, 2, 3, 0}; want != ints { // test build helper works correctly
		t.Fatalf("want %v, got %v", want, ints)
	}

	tt := []struct {
		name string
		ints []int
		want []int
	}{
		{"nil -> nil", nil, nil},
		{"3 - 2 - 1 - 0 -> 0 - 1 - 2 - 3", []int{3, 2, 1, 0}, []int{0, 1, 2, 3}},
	}

	for _, tc := range tt {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			list := buildLinkedListHelper_3_10_3(tc.ints)
			rev := reverseLinkedList_3_10_3(list)

			got := make([]int, 0, len(tc.ints))
			walkLitemsHelper_3_10_3_recurse(rev, func(l *litem_3_10_3) {
				got = append(got, l.item)
			})

			if len(tc.want) != len(got) {
				t.Fatalf("%v must be equal to %v", got, tc.want)
			}
			for i := range tc.want {
				if tc.want[i] != got[i] {
					t.Fatalf("%v must be equal to %v", got, tc.want)
				}
			}
		})
	}
}

func buildLinkedListHelper_3_10_3(ints []int) *litem_3_10_3 {
	if len(ints) == 0 {
		return nil
	}

	head := litem_3_10_3{item: ints[0]}
	tail := &head
	for _, i := range ints[1:] {
		it := litem_3_10_3{item: i}
		tail.next = &it
		tail = tail.next
	}

	return &head
}

func walkLitemsHelper_3_10_3_recurse(litem *litem_3_10_3, f func(*litem_3_10_3)) {
	if litem == nil {
		return
	}

	f(litem)
	walkLitemsHelper_3_10_3_recurse(litem.next, f)
}

// [5] Design a stack S that supports S.push(x), S.pop(), and S.findmin(), which
// returns the minimum element of S. All operations should run in constant time.

type stack_3_10_5 struct {
	s    []int
	mins []int
	n    int
}

func (s *stack_3_10_5) push(x int) {
	s.n++
	s.s = append(s.s, x)
	s.mins = append(s.mins, x)
	if s.n == 1 {
		return
	}

	if prev := s.mins[s.n-2]; prev < s.mins[s.n-1] {
		s.mins[s.n-1] = prev
	}
}

func (s *stack_3_10_5) pop() (int, bool) {
	if s.n == 0 {
		return 0, false
	}
	s.n--
	v := s.s[s.n]
	s.s = s.s[:s.n]
	s.mins = s.mins[:s.n]
	return v, true
}

func (s *stack_3_10_5) findmin() (int, bool) {
	if s.n == 0 {
		return 0, false
	}
	return s.mins[s.n-1], true
}

func TestConstantPushPopFindminStack_3_10_5(t *testing.T) {
	testeq := func(s0, s1 []int) {
		t.Helper()
		if len(s0) != len(s1) {
			t.Fatalf("non equal slices: %v and %v", s0, s1)
		}

		for i := range s0 {
			if s0[i] != s1[i] {
				t.Fatalf("non equal slices: %v and %v", s0, s1)
			}
		}
	}

	testmin := func(s *stack_3_10_5, min int) {
		// assumed _, ok := findmin is always ok == true
		t.Helper()
		v, _ := s.findmin()
		if v != min {
			t.Fatalf("expected min %d, got %d", min, v)
		}
	}

	testpop := func(s *stack_3_10_5, exp int) {
		t.Helper()
		v, _ := s.pop()
		if v != exp {
			t.Fatalf("expected %d, got %d", exp, v)
		}
	}

	var s stack_3_10_5
	testeq(s.s, nil)
	testeq(s.mins, nil)

	s.push(3)
	testmin(&s, 3)
	s.push(4)
	testmin(&s, 3)
	s.push(2)
	testmin(&s, 2)
	s.push(1000)
	testmin(&s, 2)
	s.push(-1)
	testmin(&s, -1)

	testeq(s.s, []int{3, 4, 2, 1000, -1})
	testeq(s.mins, []int{3, 3, 2, 2, -1})

	testpop(&s, -1)
	testmin(&s, 2)
	testpop(&s, 1000)
	testmin(&s, 2)
	testpop(&s, 2)
	testmin(&s, 3)
	testpop(&s, 4)
	testmin(&s, 3)
	testpop(&s, 3)
}

// https://leetcode.com/problems/count-of-smaller-numbers-after-self/
func countSmaller(ints []int) []int {
	sort.Ints(ints)
	var t *left_count_bst_e
	build_balanced_tree(ints, &t)

	smaller := make([]int, 0, len(ints))
	for _, v := range ints {
		smaller = append(smaller, delete_left_count_bst_e(&t, v, 0))
	}
	return smaller
}

type left_count_bst_e struct {
	x, count, left_tree_count int
	left, right, parent       *left_count_bst_e
	m                         map[int]*left_count_bst_e
}

func insert_left_count_bst_e_recursive(n **left_count_bst_e, p *left_count_bst_e, x int) {
	if *n == nil {
		m := map[int]*left_count_bst_e{}
		el := &left_count_bst_e{x: x, count: 1, parent: p, m: m}
		m[x] = el
		*n = el
		return
	}

	if (*n).x == x {
		(*n).count++
		return
	} else if (*n).x < x {
		insert_left_count_bst_e_recursive(&((*n).right), *n, x)
		return
	}

	(*n).left_tree_count++
	insert_left_count_bst_e_recursive(&((*n).left), *n, x)
}

func delete_left_count_bst_e(n **left_count_bst_e, x, i int) int {
	if (*n).x < x {
		return delete_left_count_bst_e((&(*n).right), x, i+(*n).count+(*n).left_tree_count)
	} else if (*n).x > x {
		(*n).left_tree_count--
		return delete_left_count_bst_e((&(*n).left), x, i)
	}

	(*n).count--
	v := i + (*n).left_tree_count
	if (*n).count > 0 {
		return v
	}

	// no child
	if (*n).left == nil && (*n).right == nil {
		*n = nil
		return v
	} else if ((*n).left != nil) && ((*n).right == nil) {
		// left child only
		(*n).left.parent = (*n).parent
		*n = (*n).left

		return v
	} else if ((*n).right != nil) && ((*n).left == nil) {
		// right child only
		(*n).right.parent = (*n).parent
		*n = (*n).right

		return v
	}

	nv := *n
	max := nv.left
	for ; max.right != nil; max = max.right {
	}
	if max.parent == nv {
		max.parent = nv.parent
		max.right = nv.right
		if nv.right != nil {
			nv.right.parent = max
		}
		*n = max
		return v
	}

	max.parent.right = max.left
	if max.left != nil {
		max.left.parent = max.parent
	}

	nv.left_tree_count -= max.count
	nv.count = max.count
	nv.x = max.x
	return v
}

func build_balanced_tree(ints []int, n **left_count_bst_e) {
	if len(ints) == 0 {
		return
	}
	m := len(ints) / 2
	el := ints[m]
	left, right := ints[:m], ints[m+1:]
	insert_left_count_bst_e_recursive(n, nil, el)
	build_balanced_tree(left, n)
	build_balanced_tree(right, n)
}
func testInvariant(n *left_count_bst_e) {
	if n == nil {
		return
	}

	v := 0
	if n.left != nil {
		v = n.left.left_tree_count + n.left.count
	}
	if v != n.left_tree_count {
		panic(fmt.Sprintf("invariant is not hold: node: %v, left: %v", n, n.left))
	}
	testInvariant(n.left)
	testInvariant(n.right)
}

func left_subtree_count_invariant(n *left_count_bst_e) int {
	if n == nil {
		return 0
	}

	c := left_subtree_count_invariant(n.left)
	if n.left_tree_count != c {
		panic(fmt.Sprintf("%v, %d", n, c))
	}

	r := n.right
	for ; r != nil; r = r.right {
		c += left_subtree_count_invariant(r)
	}
	return c + n.count
}

func TestCountSmaller(t *testing.T) {
	naiveCountSmaller := func(ints []int) []int {
		smaller := make([]int, 0, len(ints))
		for i, v := range ints {
			c := 0
			for _, v0 := range ints[i:] {
				if v0 < v {
					c++
				}
			}
			smaller = append(smaller, c)
		}

		return smaller
	}
	tt := []struct {
		intsf func(*testing.T) []int
	}{
		{func(_ *testing.T) []int { return nil }},
		{func(_ *testing.T) []int { return []int{1, 2, 3, 4, 5} }},
		{func(_ *testing.T) []int { return []int{5, 4, 3, 2, 1} }},
		{func(_ *testing.T) []int { return []int{2, 0, 1, 4} }},
		{func(_ *testing.T) []int { return []int{4, 2, 4, 0, -4} }},
		{func(t *testing.T) []int {
			t.Helper()

			b, err := ioutil.ReadFile("count_smaller_data_0.json")
			if err != nil {
				t.Fatal(err)
			}
			var ints []int
			if err := json.Unmarshal(b, &ints); err != nil {
				t.Fatal(err)
			}
			return ints
		}},
	}

	for _, tc := range tt {
		t.Run("", func(t *testing.T) {
			ints := tc.intsf(t)
			res := countSmaller(ints)
			ref := naiveCountSmaller(ints)
			if lref, lres := len(ref), len(res); lref != lres {
				t.Errorf("not eq lengths: want %d, got %d", lref, lres)
			}

			for i := range res {
				if res[i] != ref[i] {
					t.Errorf("idx %d, want %v, got %v", i, ref[i], res[i])
				}
			}
		})
	}
}
