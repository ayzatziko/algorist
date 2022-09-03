package algorist

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"math"
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

// for each e select next min and swap with current.
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

// select and elem and swap back until hit smaller element.
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

// for each char in p test wheter p[i:i+len(t)] == t, if ok ret first char index.
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

type matrix struct {
	rows, columns int
	m             [][]int
}

// no matrices equality tests since we assume they always hold.
// matrices multiplication.
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

// Chapter 3: Data Structures.

// arr is a fixed size array.
var arr [3]int

var arrdyn []int // dynamic array with size increased if no place for new element. build upon fixed size arrays.

func TestArray(t *testing.T) { arrdyn = arr[:]; fmt.Println(arrdyn) } // to avoid unused linter error.

// Chapter 3.1.2

type list_ch_3_1_2 struct {
	item int            // data item
	next *list_ch_3_1_2 // point to successor
}

// recurively test whether two lists are equal.
func lists_eq_help_3_1_2(l0, l1 *list_ch_3_1_2) bool {
	if l0 == nil && l1 == nil {
		return true
	} else if l0 == nil || l1 == nil || l0.item != l1.item {
		return false
	}
	return lists_eq_help_3_1_2(l0.next, l1.next)
}

// recursive search for an item.
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

// delete elem from list by chainging predecessor(item) next ponter to cur item next elem.
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

// recursive predecessor search.
func predecessorList_recursive_ch_3_1_2(l *list_ch_3_1_2, item int) *list_ch_3_1_2 {
	if l.next == nil {
		return nil
	} else if l.next.item == item {
		return l
	}
	return predecessorList_recursive_ch_3_1_2(l.next, item)
}

// iterative one.
func predecessorList_ch_3_1_2(l *list_ch_3_1_2, item int) *list_ch_3_1_2 {
	for ; ; l = l.next {
		if l == nil || l.next == nil {
			return nil
		}
		if l.next.item == item {
			return l
		}
	}
}

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

// Binary search tree 3.4.1
type bstitem struct {
	key string
	val int
}

type bstree struct {
	item                bstitem
	parent, left, right *bstree
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

// noticed that started to write C like style since all code in algorithm design manual written in C.

// https://leetcode.com/problems/count-of-smaller-numbers-after-self/
// Solution description https://leetcode.com/problems/count-of-smaller-numbers-after-self/discuss/2411749/Golang-solution.
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

// invirants tests below were used to find bugs, after bugfixes they are unused.
// leave even if unised since they has helped.
func test_balanced_tree_invariant(n *left_count_bst_e) {
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
	test_balanced_tree_invariant(n.left)
	test_balanced_tree_invariant(n.right)
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

// Chapter 4. Heap and heapsort

// does not guranatee left tree root less then right tree root.
type heap_4_3 []int

var make_heap_4_3 = make_heap_slow_4_3

func make_heap_slow_4_3(h *heap_4_3, ints []int) {
	if cap(*h) <= 1 {
		*h = make(heap_4_3, 1, len(ints)+1)
	} else {
		*h = (*h)[:1]
	}

	for _, v := range ints {
		heap_insert_4_3(h, v)
	}
}

// fast since making heap starts from len(h)/2 to 0.
func make_heap_fast_4_3(h *heap_4_3, ints []int) {
	n := len(ints) + 1
	if cap(*h) < n {
		*h = make(heap_4_3, n)
	} else {
		*h = (*h)[:n]
	}

	copy((*h)[1:], ints)

	for i := n / 2; i > 0; i-- {
		heap_bubble_down_recursive_4_3(h, n, i)
	}
}

func heap_insert_4_3(h *heap_4_3, n int) {
	(*h) = append((*h), n)

	if len((*h)) <= 2 {
		return
	}

	// bubble up.
	i := len((*h)) - 1
	p := i / 2
	for p >= 1 && (*h)[i] < (*h)[p] {
		(*h)[p], (*h)[i] = (*h)[i], (*h)[p]
		i, p = p, p/2
	}
}

func heap_min_4_3(h *heap_4_3) (int, bool) {
	if len(*h) == 1 {
		return 0, false
	}

	n := len(*h) - 1
	min := (*h)[1]
	(*h)[1] = (*h)[n]
	(*h) = (*h)[:n]

	heap_bubble_down_recursive_4_3(h, n, 1)
	return min, true
}

func heap_bubble_down_recursive_4_3(h *heap_4_3, n, p int) {
	mini := p * 2
	if n <= mini {
		return
	} else if mini+1 < n && (*h)[mini+1] < (*h)[mini] {
		mini = mini + 1
	}
	if (*h)[p] <= (*h)[mini] {
		return
	}
	(*h)[p], (*h)[mini] = (*h)[mini], (*h)[p]
	heap_bubble_down_recursive_4_3(h, n, mini)
}

func heapsort_4_3_(ints []int) {
	if len(ints) == 0 {
		return
	}

	var h heap_4_3
	make_heap_4_3(&h, ints)
	ints = ints[:0]
	for v, ok := heap_min_4_3(&h); ok; v, ok = heap_min_4_3(&h) {
		ints = append(ints, v)
	}
}

func TestHeapSort_4_3(t *testing.T) {
	tt := []struct {
		arr  []int
		want []int
	}{
		{[]int{}, []int{}},
		{nil, nil},
		{[]int{6, 4, 3, 2, 1}, []int{1, 2, 3, 4, 6}},
		{[]int{9, 9, 6, 6, 1, 1}, []int{1, 1, 6, 6, 9, 9}},
	}

	for _, tc := range tt {
		t.Run(fmt.Sprintf("fast:%v->%v", tc.arr, tc.want), func(t *testing.T) {
			make_heap_4_3 = make_heap_slow_4_3
			arr := make([]int, len(tc.arr))
			copy(arr, tc.arr)
			heapsort_4_3_(arr)
			for i := range tc.want {
				if arr[i] != tc.want[i] {
					t.Fatalf("i: %d, want %v, got %v", i, tc.want, arr)
				}
			}
		})
		t.Run(fmt.Sprintf("slow:%v->%v", tc.arr, tc.want), func(t *testing.T) {
			make_heap_4_3 = make_heap_fast_4_3
			arr := make([]int, len(tc.arr))
			copy(arr, tc.arr)
			heapsort_4_3_(arr)
			for i := range tc.want {
				if arr[i] != tc.want[i] {
					t.Fatalf("i: %d, want %v, got %v", i, tc.want, arr)
				}
			}
		})
	}
}

// 4.5 mergesort

type item_type_4_5 int

func merge_sort_array_4_5(arr, bufl, bufr []item_type_4_5) {
	n := len(arr)
	if n <= 1 {
		return
	}

	m := n / 2
	merge_sort_array_4_5(arr[:m], bufl, bufr)
	merge_sort_array_4_5(arr[m:], bufl, bufr)

	// arr[:m] & arr[m:] are sorted, need to merge
	copy(bufl, arr[:m])
	bufr = arr[m:]
	arr = arr[:0]

	lp, rp := 0, 0
	for lp < m && rp < n-m {
		if bufl[lp] <= bufr[rp] {
			arr = append(arr, bufl[lp])
			lp++
		} else {
			arr = append(arr, bufr[rp])
			rp++
		}
	}

	if lp < m {
		for ; lp < m; lp++ {
			arr = append(arr, bufl[lp])
		}
	}
	// else do not care about right side since values already there
}

func TestMergesort_array_4_5(t *testing.T) {
	tt := []struct {
		arr []item_type_4_5
		res []item_type_4_5
	}{
		{nil, nil},
		{[]item_type_4_5{}, []item_type_4_5{}},
		{[]item_type_4_5{5, 4, 3, 2, 1}, []item_type_4_5{1, 2, 3, 4, 5}},
		{[]item_type_4_5{5, 1, 4, 2, 3, 3, 4, 2, 1, 5}, []item_type_4_5{1, 1, 2, 2, 3, 3, 4, 4, 5, 5}},
	}

	// dont care about test names from here, writing one by one hand
	for _, tc := range tt {
		t.Run("", func(t *testing.T) {
			n := len(tc.arr)
			merge_sort_array_4_5(tc.arr, make([]item_type_4_5, n/2), make([]item_type_4_5, n/2))
			for i := range tc.res {
				if tc.arr[i] != tc.res[i] {
					t.Fatalf("want %v, got %v", tc.res, tc.arr)
				}
			}
		})
	}
}

type item_list_type_4_5 struct {
	v    item_type_4_5
	next *item_list_type_4_5
}

func merge_sort_list_4_5(head *item_list_type_4_5) *item_list_type_4_5 {
	left := head
	if left == nil || left.next == nil {
		return head
	}

	ltail, right, ahead := left, left, left
	for ahead != nil && ahead.next != nil {
		ltail = right
		right, ahead = right.next, ahead.next.next
	}
	ltail.next = nil

	lh := merge_sort_list_4_5(left)
	rh := merge_sort_list_4_5(right)

	var htmp item_list_type_4_5
	merge_recursive_4_5(&htmp, lh, rh)
	return htmp.next
}

func merge_recursive_4_5(t, l, r *item_list_type_4_5) {
	if l == nil {
		t.next = r
		return
	} else if r == nil {
		t.next = l
		return
	}

	if l.v <= r.v {
		t.next = l
		merge_recursive_4_5(t.next, l.next, r)
		return
	}

	t.next = r
	merge_recursive_4_5(t.next, l, r.next)
}

func build_list_4_5(vv []item_type_4_5) *item_list_type_4_5 {
	if len(vv) == 0 {
		return nil
	}

	n := len(vv)
	var tail *item_list_type_4_5

	for i := n - 1; i >= 0; i-- {
		tail = &item_list_type_4_5{v: vv[i], next: tail}
	}

	return tail
}

func list_4_5_eq(a, b *item_list_type_4_5) bool {
	if a == nil && b == nil {
		return true
	} else if (a == nil) != (b == nil) {
		return false
	} else if a.v != b.v {
		return false
	}

	return list_4_5_eq(a.next, b.next)
}

func TestMergesort_list_5_5(t *testing.T) {
	tt := []struct {
		arr []item_type_4_5
		res []item_type_4_5
	}{
		{nil, nil},
		{[]item_type_4_5{}, []item_type_4_5{}},
		{[]item_type_4_5{5, 4, 3, 2, 1}, []item_type_4_5{1, 2, 3, 4, 5}},
		{[]item_type_4_5{5, 1, 4, 2, 3, 3, 4, 2, 1, 5}, []item_type_4_5{1, 1, 2, 2, 3, 3, 4, 4, 5, 5}},
	}

	// dont care about test names from here, writing by one hand
	for _, tc := range tt {
		t.Run("", func(t *testing.T) {
			h := merge_sort_list_4_5(build_list_4_5(tc.arr))
			if w := build_list_4_5(tc.res); !list_4_5_eq(h, w) {
				t.Fatalf("%#v, %#v", w, h)
			}
		})
	}
}

func quicksort_4_6(arr []item_type_4_5) {
	n := len(arr)
	if n <= 1 {
		return
	}

	fhigh, piv := 0, arr[n-1]
	for i := 0; i < n-1; i++ {
		if arr[i] <= piv {
			arr[fhigh], arr[i] = arr[i], arr[fhigh]
			fhigh++
		}
	}

	arr[fhigh], arr[n-1] = arr[n-1], arr[fhigh]
	quicksort_4_6(arr[:fhigh])
	quicksort_4_6(arr[fhigh+1:])
}

func TestQuicksort_4_6(t *testing.T) {
	tt := []struct {
		arr []item_type_4_5
		res []item_type_4_5
	}{
		{nil, nil},
		{[]item_type_4_5{}, []item_type_4_5{}},
		{[]item_type_4_5{5, 4, 3, 2, 1}, []item_type_4_5{1, 2, 3, 4, 5}},
		{[]item_type_4_5{5, 1, 4, 2, 3, 3, 4, 2, 1, 5}, []item_type_4_5{1, 1, 2, 2, 3, 3, 4, 4, 5, 5}},
	}

	// dont care about test names from here, writing by one hand
	for _, tc := range tt {
		t.Run("", func(t *testing.T) {
			quicksort_4_6(tc.arr)

			for i := range tc.res {
				if tc.arr[i] != tc.res[i] {
					t.Fatalf("want %v, got %v", tc.res, tc.arr)
				}
			}
		})
	}
}

// 5. Divide and conquer

// https://leetcode.com/problems/median-of-two-sorted-arrays/submissions/
func findMedianSortedArrays(nums1 []int, nums2 []int) float64 {
	n := len(nums1) + len(nums2)
	t := make([]int, 0, n)
	for len(nums1) > 0 && len(nums2) > 0 {
		if nums1[0] < nums2[0] {
			t = append(t, nums1[0])
			nums1 = nums1[1:]
		} else {
			t = append(t, nums2[0])
			nums2 = nums2[1:]
		}
	}

	t = append(t, nums1...)
	t = append(t, nums2...)

	if n%2 == 1 {
		return float64(t[n/2])
	}

	sum := t[n/2] + t[n/2-1]
	return float64(sum) / float64(2)
}

func TestFindMedianSortedArrays(t *testing.T) {
	tt := []struct {
		nums1, nums2 []int
		out          float64
	}{
		{[]int{1, 3}, []int{2}, 2},
		{[]int{1, 2}, []int{3, 4}, 2.5},
	}

	for _, tc := range tt {
		t.Run("", func(t *testing.T) {
			o := findMedianSortedArrays(tc.nums1, tc.nums2)
			if o != tc.out {
				t.Fatalf("unexpected median: want %.1f, got %.1f", tc.out, o)
			}
		})
	}
}

// https://leetcode.com/problems/maximum-subarray/
// not solved :(
func maxSubArray(nums []int) int {
	s, _, _ := maxSubArrayRecursive(nums, 0, len(nums))
	return s
}

func maxSubArrayRecursive(nums []int, low, high int) (int, int, int) {
	if high-low == 1 {
		return nums[low], low, high
	}

	m := (high + low) / 2
	left, ll, lh := maxSubArrayRecursive(nums, low, m)
	right, rl, rh := maxSubArrayRecursive(nums, m, high)

	s := left + right
	for i := lh; i < rl; i++ {
		s += nums[i]
	}
	if s > left && s > right {
		return s, ll, rh
	}

	if left > right {
		return left, ll, lh
	}

	return right, rl, rh
}

func TestMaxSubArray(t *testing.T) {
	tt := []struct {
		nums []int
		out  int
	}{
		{[]int{-2, 1, -3, 4, -1, 2, 1, -5, 4}, 6},
		{[]int{1}, 1},
		{[]int{5, 4, -1, 7, 8}, 23},
	}

	for _, tc := range tt {
		t.Run("", func(t *testing.T) {
			o := maxSubArray(tc.nums)
			if o != tc.out {
				t.Fatalf("unexpected median: want %d, got %d", tc.out, o)
			}
		})
	}
}

// 7. Graphs

type edgenode struct {
	y      int
	weight int
	next   *edgenode
}

const maxverticies = 100

type graph_7 struct {
	edges      []*edgenode // starts from 1
	degree     []int
	nverticies int
	nedges     int
	directed   bool
}

func read_graph_7(g *graph_7, directed bool, b []byte) error {
	if g.degree == nil {
		g.degree = make([]int, maxverticies+1)
	}
	for i := range g.degree {
		g.degree[i] = 0
	}
	if g.edges == nil {
		g.edges = make([]*edgenode, maxverticies+1)
	}
	for i := range g.edges {
		g.edges[i] = nil
	}
	g.nedges = 0
	g.nverticies = 0
	g.directed = directed

	s := bufio.NewScanner(bytes.NewBuffer(b))
	if !s.Scan() {
		return nil
	}
	if _, err := fmt.Fscanf(bytes.NewBuffer(s.Bytes()), "%d", &g.nverticies); err != nil {
		return err
	}

	var x, y int
	for s.Scan() {
		if _, err := fmt.Fscanf(bytes.NewBuffer(s.Bytes()), "%d %d", &x, &y); err != nil {
			return err
		}

		insert_edge(g, x, y, directed)
	}

	return nil
}

func insert_edge(g *graph_7, x, y int, directed bool) {
	e := edgenode{y: y, next: g.edges[x]}
	g.edges[x] = &e
	g.degree[x]++
	if !directed {
		insert_edge(g, y, x, true)
	} else {
		g.nedges++
	}
}

func string_graph(g *graph_7) string {
	var b bytes.Buffer
	var e *edgenode
	for i := 1; i <= g.nverticies; i++ {
		fmt.Fprintf(&b, "%d:", i)
		for e = g.edges[i]; e != nil; e = e.next {
			fmt.Fprintf(&b, " %d", e.y)
		}
		fmt.Fprintf(&b, "\n")
	}
	return b.String()
}

func TestGraph_7_2(t *testing.T) {
	tt := []struct {
		in       string
		directed bool
		out      string
	}{
		{"", false, ""},
		{"4\n1 2\n2 3\n2 4\n3 4\n", true, "1: 2\n2: 4 3\n3: 4\n4:\n"},
		{"4\n1 2\n2 3\n2 4\n3 4\n", false, "1: 2\n2: 4 3 1\n3: 4 2\n4: 3 2\n"},
	}

	var g graph_7
	for _, tc := range tt {
		t.Run("", func(t *testing.T) {
			read_graph_7(&g, tc.directed, []byte(tc.in))
			if s := string_graph(&g); s != tc.out {
				t.Fatalf("unexpected output: \nwant %s\ngot %s", tc.out, s)
			}
		})
	}
}

// dont bother about vars globality, in real code they all will be used as arguments.
var (
	graph_discovered_7 = make([]bool, maxverticies+1)
	graph_processed_7  = make([]bool, maxverticies+1)
	graph_parents_7    = make([]int, maxverticies+1)
	graph_time_7       int
)

func graph_init_search_7(g *graph_7) {
	for i := 0; i <= g.nverticies; i++ {
		graph_discovered_7[i] = false
		graph_processed_7[i] = false
	}
	graph_time_7 = 0
}

// insert start el in queue.
// for each vertex in queue mark current el discovered, insert all undiscovered elements in queue, process until queue not empty.
func graph_bfs_7_6(g *graph_7, s int,
	process_vertex_before, process_vertex_after func(int),
	process_edge func(int, int),
) {

	queue := make([]int, 1, g.nverticies)
	queue[0] = s // at the beggining only 1 element in queue.
	graph_parents_7[s] = -1

	for len(queue) > 0 {
		v := queue[0]
		queue = queue[1:]

		graph_discovered_7[v] = true
		process_vertex_before(v)

		for p := g.edges[v]; p != nil; p = p.next {
			if !graph_processed_7[p.y] || g.directed {
				process_edge(v, p.y)
			}
			if !graph_discovered_7[p.y] {
				queue = append(queue, p.y)
				graph_discovered_7[p.y] = true
				graph_parents_7[p.y] = v
			}
		}
		graph_processed_7[v] = true
		process_vertex_after(v)
	}
}

func TestGraph_bfs_7_6(t *testing.T) {
	tt := []struct {
		in       string
		directed bool
		s        int

		out string // a->b\n
	}{
		{"4\n1 2\n2 3\n2 4\n3 4\n", true, 1, "1->2\n2->4\n2->3\n3->4\n"},
		{"4\n1 2\n2 3\n2 4\n3 4\n", false, 1, "1->2\n2->4\n2->3\n4->3\n"},
	}

	var g graph_7
	for _, tc := range tt {
		t.Run("", func(t *testing.T) {
			read_graph_7(&g, tc.directed, []byte(tc.in))
			s := ""
			graph_init_search_7(&g)
			graph_bfs_7_6(&g, tc.s, func(i int) {}, func(i int) {}, func(a, b int) {
				s += fmt.Sprintf("%d->%d\n", a, b)
			})

			if s != tc.out {
				t.Fatalf("unexpected output: \nwant %s\ngot %s", tc.out, s)
			}
		})
	}
}

// build path from parents array by pointing back from s until we hit e.
func graph_bfs_find_path_7_6(s, e int, parents []int) []int {
	path := graph_bfs_find_path_recursive_7_6(s, e, parents, nil)
	for i, e := 0, len(path)-1; i < e; i, e = i+1, e-1 {
		path[i], path[e] = path[e], path[i]
	}
	return path
}

func graph_bfs_find_path_recursive_7_6(s, e int, parents, path []int) []int {
	if s == e || parents[e] == -1 {
		return append(path, e)
	}

	return graph_bfs_find_path_recursive_7_6(s, parents[e], parents, append(path, e))
}

func TestGraph_bfs_find_path_7_6(t *testing.T) {
	tt := []struct {
		in       string
		directed bool
		start    int
		from, to int

		out []int // {from, ..., end}
	}{
		{"4\n1 2\n2 3\n2 4\n3 4\n", true, 1, 1, 4, []int{1, 2, 4}},
		{"4\n1 2\n2 3\n2 4\n3 4\n", false, 1, 1, 4, []int{1, 2, 4}},
	}

	var g graph_7
	for _, tc := range tt {
		t.Run("", func(t *testing.T) {
			read_graph_7(&g, tc.directed, []byte(tc.in))
			graph_init_search_7(&g)
			graph_bfs_7_6(&g, tc.start, func(i int) {}, func(i int) {}, func(a, b int) {})
			path := graph_bfs_find_path_7_6(tc.from, tc.to, graph_parents_7)
			for i := 0; i < len(path); i++ {
				if path[i] != tc.out[i] {
					t.Fatalf("unexpected output: \nwant %v\ngot %v", tc.out, path)
				}
			}
		})
	}
}

// components connected if v0 from component 1 have a path to v1 from component2.
// bfs from s adds only reachable vericies from s, so all added verticies belongs to 1 component.
// we itereate over all verticies, all undisovered verticies are considered to be a separated component.
func graph_connected_components_7_7(g *graph_7) []int {
	var components []int
	for i := 1; i <= g.nverticies; i++ {
		if !graph_discovered_7[i] {
			graph_bfs_7_6(g, i, func(i int) {}, func(i int) {}, func(_, _ int) {})
			components = append(components, i)
		}
	}
	return components
}

func TestGraph_connected_components_7_6(t *testing.T) {
	tt := []struct {
		in       string
		directed bool

		out []int // {c0, ..., cN}
	}{
		{"6\n1 2\n2 3\n2 4\n3 4\n5 6", true, []int{1, 5}},
	}

	var g graph_7
	for _, tc := range tt {
		t.Run("", func(t *testing.T) {
			read_graph_7(&g, tc.directed, []byte(tc.in))
			graph_init_search_7(&g)
			cc := graph_connected_components_7_7(&g)
			for i := 0; i < len(cc); i++ {
				if cc[i] != tc.out[i] {
					t.Fatalf("unexpected output: \nwant %v\ngot %v", tc.out, cc)
				}
			}
		})
	}
}

const (
	graph_white = -1
	graph_black = 1
)

var (
	graph_colors_7_7    = make([]int, maxverticies+1)
	graph_bipartite_7_7 = false
)

// partners a man and woman, woman cannot has a woman partner, man cannot have a man partner.
// graph is bipartite if we can color verticies by an opposite color of parent.
// if in case we changing a already colored v to other color, than graph is not bipartite.
func graph_two_color_7_7(g *graph_7) {
	graph_bipartite_7_7 = true
	for i := range graph_colors_7_7 {
		graph_colors_7_7[i] = 0
	}

	for i := 1; i <= g.nverticies; i++ {
		if !graph_discovered_7[i] {
			graph_colors_7_7[i] = graph_white
			graph_bfs_7_6(g, i, func(i int) {}, func(i int) {}, func(a, b int) {
				if graph_colors_7_7[a] == graph_colors_7_7[b] {
					graph_bipartite_7_7 = false
				}

				graph_colors_7_7[b] = -1 * graph_colors_7_7[a]
			})
		}
	}
}

func TestGraph_two_color_7_7(t *testing.T) {
	tt := []struct {
		in       string
		directed bool

		bipartite bool
	}{
		{"6\n1 2\n2 3\n2 4\n5 6", true, true},
		{"6\n1 2\n2 3\n3 2\n2 4\n5 6", false, true},
		{"6\n1 2\n2 3\n2 4\n3 1\n5 6", true, false},
		{"6\n1 2\n2 3\n2 4\n3 1\n5 6", false, false},
	}

	var g graph_7
	for _, tc := range tt {
		t.Run("", func(t *testing.T) {
			read_graph_7(&g, tc.directed, []byte(tc.in))
			graph_init_search_7(&g)
			graph_two_color_7_7(&g)
			if graph_bipartite_7_7 != tc.bipartite {
				t.Fatalf("unexpected output: \nwant %v\ngot %v", tc.bipartite, graph_bipartite_7_7)
			}
		})
	}
}

var (
	// entry/exit times are needed to correctly identify edge type, see graph edge classification func.
	graph_entry_time_7_7 = make([]int, maxverticies+1)
	graph_exit_time_7_7  = make([]int, maxverticies+1)

	// to faster exit from dfs.
	graph_dfs_finished = false
)

// dfs is based on stack which is naturally implemented by call stack.
// mark current v as entered and discovered, for each adjacent edge (if we can) do a dfs.
// after processing all adjacent edges mark current is processed. Child nodes are processed earlier than parent node.
func graph_dfs_7_8(g *graph_7, s int,
	process_vertex_before, process_vertex_after func(int),
	process_edge func(int, int)) {

	graph_discovered_7[s] = true
	graph_time_7++
	graph_entry_time_7_7[s] = graph_time_7
	process_vertex_before(s)

	for p := g.edges[s]; p != nil; p = p.next {
		if !graph_discovered_7[p.y] {
			graph_parents_7[p.y] = s
			process_edge(s, p.y)
			graph_dfs_7_8(g, p.y, process_vertex_before, process_vertex_after, process_edge)
		} else if (!graph_processed_7[p.y] && graph_parents_7[s] != p.y) || g.directed {
			process_edge(s, p.y)
		}

		if graph_dfs_finished {
			return
		}
	}

	graph_time_7++
	graph_exit_time_7_7[s] = graph_time_7
	graph_processed_7[s] = true
}

func TestGraph_dfs_7_8(t *testing.T) {
	tt := []struct {
		in       string
		directed bool
		s        int

		out string // a->b\n
	}{
		{"4\n1 2\n2 3\n2 4\n3 4\n", true, 1, "1->2\n2->4\n2->3\n3->4\n"},
		{"4\n1 2\n2 3\n2 4\n3 4\n", false, 1, "1->2\n2->4\n4->3\n3->2\n"},
	}

	var g graph_7
	for _, tc := range tt {
		t.Run("", func(t *testing.T) {
			read_graph_7(&g, tc.directed, []byte(tc.in))
			s := ""
			graph_init_search_7(&g)
			graph_dfs_7_8(&g, tc.s, func(i int) {}, func(i int) {}, func(a, b int) {
				s += fmt.Sprintf("%d->%d\n", a, b)
			})

			if s != tc.out {
				t.Fatalf("unexpected output: \nwant %s\ngot %s", tc.out, s)
			}
		})
	}
}

// Here is the place for procedures of finding articulation verticies, but it was hard to understand for now.
// Will return to it later.

const (
	edge_back    = 0
	edge_tree    = 1
	edge_forward = 2
	edge_cross   = 3
)

func graph_edge_classification_7(x, y int) (int, bool) {
	switch {
	case graph_parents_7[y] == x:
		return edge_tree, true // parent points to child
	case graph_discovered_7[y] && !graph_processed_7[y]:
		return edge_back, true // y is in discovered but not processed, than it is above in call stack, than this edge points back.
	case graph_processed_7[y] && graph_entry_time_7_7[y] > graph_entry_time_7_7[x]:
		return edge_forward, true // already processed by one of x parent's childs, this edge is like a teleport forward.
	case graph_processed_7[y] && graph_entry_time_7_7[y] < graph_entry_time_7_7[x]:
		return edge_cross, true // processed by one of x's parent's child
	}

	fmt.Printf("Warning: self loop (%d,%d)\n", x, y)
	return -1, false
}

// top sort is based on idea that last processed item is a start point to process earlier appeared items.
// so append in process edge laster and reverse array.
// looks a bit like a connected component algo, idea behind that is node is not discovered then there are no
// path from discovered node to undiscovered, but path from undiscovered to discovered must exist. Then undiscovered
// node must appear earlier in topsort.
func graph_topsort_7_10(g *graph_7) ([]int, bool) {
	s := make([]int, 0, g.nverticies)
	dag := true
	for i := 1; i <= g.nverticies && !graph_dfs_finished; i++ {
		if !graph_discovered_7[i] {
			graph_dfs_7_8(g, i, func(_ int) {}, func(x int) {
				s = append(s, x)
			}, func(a, b int) {
				if edgetyp, ok := graph_edge_classification_7(a, b); !ok || edgetyp == edge_back {
					dag = false
					graph_dfs_finished = true
				}
			})
		}
	}

	for i, n := 0, len(s); i > n; i, n = i+1, n+1 {
		s[i], s[n] = s[n], s[i]
	}
	return s, dag
}

func TestGraph_topsort_7_10(t *testing.T) {
	tt := []struct {
		in       string
		directed bool

		out   []int
		outok bool
	}{
		{"4\n1 2\n2 3\n2 4\n3 4\n", true, []int{4, 3, 2, 1}, true},
		{"4\n2 1\n2 4\n3 2\n3 4\n", true, []int{3, 2, 4, 1}, true},
		{"4\n1 2\n2 3\n2 4\n3 4\n4 1\n", true, nil, false}, // backedge
	}

	var g graph_7
	for _, tc := range tt {
		t.Run("", func(t *testing.T) {
			read_graph_7(&g, tc.directed, []byte(tc.in))
			graph_init_search_7(&g)
			o, ok := graph_topsort_7_10(&g)

			for i := range o {
				if o[i] != tc.out[i] {
					t.Fatalf("unexpected output: \nwant %v,%v\ngot %v,%v", tc.out, tc.outok, o, ok)
				}
			}

			if ok != tc.outok {
				t.Fatalf("unexpected output: \nwant %v,%v\ngot %v,%v", tc.out, tc.outok, o, ok)
			}
		})
	}
}

// a bit of explaination to myself.
// For ex we have a graph
// 6\n1 3\n1 2\n2 3\n2 5\n3 4\n4 5\n5 6\n 6 4\n
// in this graph we have a 2 strongly connected components [1 2 3] and [4 5 6].
// The trick for is that it is really hard to see that code really does the work,
// but a have written an execution on papaer for a reversed graph and this is actually works.
// For this graph a transposed graph really has not outgoing edges from [1 2 3] to [4 5 6].
func graph_strong_components_7_10(g *graph_7) [][]int {
	// consider a graph from comment avove.

	dfs1stack := make([]int, 0, g.nverticies)

	for i := 1; i < g.nverticies; i++ {
		if !graph_discovered_7[i] {
			graph_dfs_7_8(g, i, func(_ int) {}, func(x int) {
				dfs1stack = append(dfs1stack, x)
			}, func(_, _ int) {})
		}
	}
	// at the end of the day dfs1stack for a graph above should be 4,6,5,3,2,1

	// build a transpose graph - the same graph but verticies reversed.
	// traverse such graph gives us a nodes can reach any node.
	gt := graph_7{
		nverticies: g.nverticies,
		directed:   true,
	}
	for i := 1; i < g.nverticies; i++ {
		for p := g.edges[i]; p != nil; p = p.next {
			insert_edge(g, p.y, i, true)
		}
	}
	// a transposed graph is 6\n3 1\n2 1\n3 2\n5 2\n4 3\n5 4\n6 5\n4 6\n

	var components [][]int
	var cur []int
	graph_init_search_7(&gt)
	for len(dfs1stack) > 0 {
		i := dfs1stack[len(dfs1stack)-1]
		dfs1stack = dfs1stack[:len(dfs1stack)-1]
		if !graph_discovered_7[i] {
			components = append(components, cur)
			cur = nil
			graph_dfs_7_8(&gt, i, func(x int) {
				cur = append(cur, x)
			}, func(_ int) {}, func(_, _ int) {})
		}
	}

	return components
}

func TestGraph_strong_components_7_10(t *testing.T) {
	tt := []struct {
		in       string
		directed bool

		out [][]int
	}{
		{"8\n1 2\n2 3\n2 4\n2 5\n3 1\n4 1\n4 6\n4 8\n8 6\n6 7\n7 5\n5 6\n", true, [][]int{{1, 2, 3, 4}, {5, 6, 7}, {8}}},
		{"6\n1 3\n1 2\n2 3\n2 5\n3 4\n4 5\n5 6\n 6 4\n", true, [][]int{{1, 2, 3}, {4, 5, 6}}},
	}

	var g graph_7
	for _, tc := range tt {
		t.Run("", func(t *testing.T) {
			read_graph_7(&g, tc.directed, []byte(tc.in))
			graph_init_search_7(&g)
			components := graph_strong_components_7_10(&g)

			for _, v := range components {
				sort.Ints(v)
			}
			sort.Slice(components, func(i, j int) bool {
				return components[i][0] < components[j][0]
			})

			for i, c := range components {
				for j := range c {
					if c[j] != tc.out[i][j] {
						t.Fatalf("unexpected components: \nwant %v\ngot %v", tc.out, components)
					}
				}
			}
		})
	}
}

// https://leetcode.com/problems/minimum-height-trees/
// cant solve for now:(
func findMinHeightTrees(n int, rawedges [][]int) []int {
	return nil
}

func Test_findMinHeightTrees(t *testing.T) {
	tt := []struct {
		n     int
		edges [][]int

		out []int
	}{
		{4, [][]int{{1, 0}, {1, 2}, {1, 3}}, []int{1}},
		{6, [][]int{{3, 0}, {3, 1}, {3, 2}, {3, 4}, {5, 4}}, []int{3, 4}},
		{1, [][]int{}, []int{0}},
		{2, [][]int{{0, 1}}, []int{0, 1}},
	}

	for _, tc := range tt {
		t.Run("", func(t *testing.T) {
			o := findMinHeightTrees(tc.n, tc.edges)
			if len(o) != len(tc.out) {
				t.Fatalf("unexpected output: want %v, got %v", tc.out, o)
			}

			for i, v := range o {
				if v != tc.out[i] {
					t.Fatalf("unexpected output: want %v, got %v", tc.out, o)
				}
			}
		})
	}
}

// https://leetcode.com/problems/redundant-connection/
func findRedundantConnection(edges [][]int) []int {
	// self designed union find ds. implemented properely later.
	sets := map[int][]int{}
	setheads := map[int]int{}
	add := func(e []int) bool {
		h0, ok0 := setheads[e[0]]
		h1, ok1 := setheads[e[1]]
		if !ok0 && !ok1 { // no edges in any set, create one
			sets[e[0]] = append(sets[e[0]], e[0], e[1])
			setheads[e[0]] = e[0]
			setheads[e[1]] = e[0]
			return false
		} else if ok0 && !ok1 { // first v in set, add to existing
			sets[h0] = append(sets[h0], e[1])
			setheads[e[1]] = h0
			return false
		} else if !ok0 && ok1 {
			sets[h1] = append(sets[h1], e[0])
			setheads[e[0]] = h1
			return false
		} else if h0 != h1 {
			// connect graphs by new edge, still tree, merge sets
			for _, v := range sets[h1] {
				setheads[v] = h0
			}
			sets[h0] = append(sets[h0], sets[h1]...)
			delete(sets, h1)
			return false
		}

		// already in the same set
		return true
	}

	m := -1
	for i, e := range edges {
		if add(e) {
			m = i
		}
	}

	return edges[m]
}

func Test_findRedundantConnection(t *testing.T) {
	tt := []struct {
		edges [][]int
		r     []int
	}{
		{[][]int{{1, 2}, {1, 3}, {2, 3}}, []int{2, 3}},
		{[][]int{{1, 2}, {2, 3}, {3, 4}, {1, 4}, {1, 5}}, []int{1, 4}},
		{[][]int{{1, 5}, {3, 4}, {3, 5}, {4, 5}, {2, 4}}, []int{4, 5}},
	}

	for _, tc := range tt {
		t.Run("", func(t *testing.T) {
			r := findRedundantConnection(tc.edges)
			if r[0] != tc.r[0] || r[1] != tc.r[1] {
				t.Fatalf("unexpected redundant edge: want %v, got %v", tc.r, r)
			}
		})
	}
}

// Chapter 8. Weighted graphs

func read_weighted_graph_8(g *graph_7, directed bool, b []byte) error {
	if g.degree == nil {
		g.degree = make([]int, maxverticies+1)
	}
	for i := range g.degree {
		g.degree[i] = 0
	}
	if g.edges == nil {
		g.edges = make([]*edgenode, maxverticies+1)
	}
	for i := range g.edges {
		g.edges[i] = nil
	}
	g.nedges = 0
	g.nverticies = 0
	g.directed = directed

	s := bufio.NewScanner(bytes.NewBuffer(b))
	if !s.Scan() {
		return nil
	}
	if _, err := fmt.Fscanf(bytes.NewBuffer(s.Bytes()), "%d", &g.nverticies); err != nil {
		return err
	}

	var x, y, weight int
	for s.Scan() {
		if _, err := fmt.Fscanf(bytes.NewBuffer(s.Bytes()), "%d %d %d", &x, &y, &weight); err != nil {
			return err
		}

		insert_edge_weighted_8(g, x, y, weight, directed)
	}

	return nil
}

func insert_edge_weighted_8(g *graph_7, x, y, weight int, directed bool) {
	e := edgenode{y: y, weight: weight, next: g.edges[x]}
	g.edges[x] = &e
	g.degree[x]++
	if !directed {
		insert_edge_weighted_8(g, y, x, weight, true)
	} else {
		g.nedges++
	}
}

// mst stands for minimum spanning tree
// start point has a weight 0.
// until current processing item not in tree, add to tree and to a total weight,
// for current item update distances to edges, select a smallest one and set current item to a smallest one.
// exists in case of for current el all adjacent nodes are inserted so v stays unchaged.
func prim_mst_8(g *graph_7, start int) int {
	intree := make([]bool, g.nverticies+1)
	distance := make([]int, g.nverticies+1)
	for i := 1; i <= g.nverticies; i++ {
		distance[i] = math.MaxInt
		graph_parents_7[i] = -1
	}
	total := 0
	dist := 0

	v := start
	for !intree[v] {
		intree[v] = true
		if v != start {
			fmt.Printf("prim_mst_8: adding edge (%d, %d) with weight %d\n", graph_parents_7[v], v, distance[v])
			total += distance[v]
		}

		for p := g.edges[v]; p != nil; p = p.next {
			if !intree[p.y] && distance[p.y] > p.weight {
				distance[p.y] = p.weight
				graph_parents_7[p.y] = v
			}
		}

		dist = math.MaxInt
		for i := 1; i <= g.nverticies; i++ {
			if !intree[i] && dist > distance[i] {
				dist = distance[i]
				v = i
			}
		}
	}

	return total
}

func TestWeightedGraph_prim_mst_8(t *testing.T) {
	tt := []struct {
		in       string
		directed bool

		out int
	}{
		{"7\n7 1 5\n7 2 7\n7 3 9\n1 3 7\n2 3 4\n1 5 12\n5 3 4\n5 6 7\n6 3 3\n6 2 2\n2 4 5\n4 6 2\n",
			false, 23},
	}

	var g graph_7
	for _, tc := range tt {
		t.Run("", func(t *testing.T) {
			read_weighted_graph_8(&g, tc.directed, []byte(tc.in))
			graph_init_search_7(&g)
			w := prim_mst_8(&g, 7)
			if w != tc.out {
				t.Fatalf("unexpected output: want %d, got %d", tc.out, w)
			}
		})
	}
}

// kruskal algo idea is to peek smallest by weight not inserted edge, and if verticies
// of edge is not in the same component merge components and build by this procedure
// minimum spanning tree.
func kruskal_mst_8(g *graph_7) int {
	wedges := make([]weighted_edge_8, 0, g.nedges)
	for x, p := range g.edges {
		for ; p != nil; p = p.next {
			wedges = append(wedges, weighted_edge_8{x, p.y, p.weight})
		}
	}
	quicksort_edges_kruskal(wedges)

	var set union_find_8
	init_union_find_8(&set, g.nverticies)

	total := 0
	for _, we := range wedges {
		if !same_component_8(&set, we.x, we.y) {
			fmt.Printf("kruskal: adding edge(%d, %d, %d) into mst", we.x, we.y, we.weight)
			union_sets_8(&set, we.x, we.y)
			total += we.weight
		}
	}

	return total
}

type weighted_edge_8 struct {
	x, y, weight int
}

func quicksort_edges_kruskal(wedges []weighted_edge_8) {
	if len(wedges) <= 1 {
		return
	}

	last := len(wedges) - 1
	p := wedges[last]
	firsthigh := 0

	for i := 0; i < last; i++ {
		if wedges[i].weight < p.weight {
			wedges[firsthigh], wedges[i] = wedges[i], wedges[firsthigh]
			firsthigh++
		}
	}

	wedges[firsthigh], wedges[last] = wedges[last], wedges[firsthigh]
	quicksort_edges_kruskal(wedges[:firsthigh])
	quicksort_edges_kruskal(wedges[firsthigh+1:])
}

// I already implemented union_find data structure by myself, this is the other one.
// union find data structure is needed to fast comparison whether 2 verticies belongs
// to the same component or not and constant components merge.

const set_size = 100

type union_find_8 struct {
	parent []int
	size   []int
	n      int
}

func init_union_find_8(s *union_find_8, n int) {
	s.n = n
	if s.size == nil {
		s.size = make([]int, set_size+1)
	}
	if s.parent == nil {
		s.parent = make([]int, set_size+1)
	}
	for i := 1; i <= n+1; i++ {
		s.size[i] = 1
		s.parent[i] = i
	}
}

func find_8(s *union_find_8, x int) int {
	if p := s.parent[x]; p != x {
		return find_8(s, p)
	}
	return x
}

func union_sets_8(s *union_find_8, x, y int) {
	r1, r2 := find_8(s, x), find_8(s, y)
	if r1 == r2 {
		return // already in the same set
	}

	// merge smaller tree into bigger one to not increase tree height
	if s.size[r1] < s.size[r2] {
		s.size[r2] += s.size[r1]
		s.parent[r1] = r2
		return
	}

	s.size[r1] += s.size[r2]
	s.parent[r2] = r1
}

func same_component_8(s *union_find_8, x, y int) bool {
	return find_8(s, x) == find_8(s, y)
}

func TestWeightedGraph_prim_mht_8(t *testing.T) {
	tt := []struct {
		in       string
		directed bool

		out int
	}{
		{"7\n7 1 5\n7 2 7\n7 3 9\n1 3 7\n2 3 4\n1 5 12\n5 3 4\n5 6 7\n6 3 3\n6 2 2\n2 4 5\n4 6 2\n",
			false, 23},
	}

	var g graph_7
	for _, tc := range tt {
		t.Run("", func(t *testing.T) {
			read_weighted_graph_8(&g, tc.directed, []byte(tc.in))
			graph_init_search_7(&g)
			w := kruskal_mst_8(&g)
			if w != tc.out {
				t.Fatalf("unexpected output: want %d, got %d", tc.out, w)
			}
		})
	}
}

// related probles to mst
// max span tree: find by prim/kruskal by negating weights.
// min product span tree: didnot understand about weight products, but to use algo to find min prod st is replace weight by log(weight)

// same as mst prim, but update distance to new node each time new distance (distance to v + y edge weight) less than old distance to y.
// for each elem not in tree take element with less distance and process it.
// ends when all edges a added and v is not updated.
func graph_dijkstra_shortest_path_8(g *graph_7, s int) (int, []int) {
	intree := make([]bool, g.nverticies+1)
	distance := make([]int, g.nverticies+1)
	for i := 1; i <= g.nverticies; i++ {
		distance[i] = math.MaxInt
		graph_parents_7[i] = -1
	}
	distance[s] = 0
	weight := 0
	v := s
	for !intree[v] {
		intree[v] = true
		if v != s {
			fmt.Printf("dijkstra shortest path: add edge (%d, %d, %d)\n", graph_parents_7[v], v, distance[v])
			weight += distance[v]
		}

		for p := g.edges[v]; p != nil; p = p.next {
			if distance[p.y] > distance[v]+p.weight {
				distance[p.y] = distance[v] + p.weight
				graph_parents_7[p.y] = v
			}
		}

		dist := math.MaxInt
		for i := 1; i <= g.nverticies; i++ {
			if !intree[i] && distance[i] < dist {
				v = i
				dist = distance[i]
			}
		}
	}

	return weight, distance
}

func TestWeightedGraph_dikkstra_shortest_path_8(t *testing.T) {
	tt := []struct {
		in       string
		directed bool

		s, t, dist int
	}{
		{"7\n1 2 5\n2 3 7\n1 3 9\n1 4 7\n4 3 4\n2 5 12\n5 3 4\n5 6 7\n6 3 3\n4 6 2\n4 7 5\n7 6 2\n",
			false, 1, 7, 11},
		{"4\n1 2 1\n1 3 2\n2 4 10\n3 4 2", false, 1, 4, 4},
	}

	var g graph_7
	for _, tc := range tt {
		t.Run("", func(t *testing.T) {
			read_weighted_graph_8(&g, tc.directed, []byte(tc.in))
			graph_init_search_7(&g)
			_, dist := graph_dijkstra_shortest_path_8(&g, tc.s)
			if dist[tc.t] != tc.dist {
				t.Fatalf("unexpected output: want %d, got %d", tc.dist, dist[tc.t])
			}
		})
	}
}
