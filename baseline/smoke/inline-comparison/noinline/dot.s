	.file	"dot.c"
	.option nopic
	.attribute arch, "rv64i2p1_m2p0_c2p0_zmmul1p0_zca1p0"
	.attribute unaligned_access, 0
	.attribute stack_align, 16
	.text
	.align	1
	.globl	dot8_t
	.type	dot8_t, @function
dot8_t:
	ble	a2,zero,.L4
	mv	a5,a0
	add	a2,a0,a2
	li	a0,0
.L3:
	lb	a4,0(a5)
	lb	a3,0(a1)
	addi	a5,a5,1
	addi	a1,a1,1
	mulw	a4,a4,a3
	addw	a0,a4,a0
	bne	a2,a5,.L3
	ret
.L4:
	li	a0,0
	ret
	.size	dot8_t, .-dot8_t
	.section	.text.startup,"ax",@progbits
	.align	1
	.globl	main
	.type	main, @function
main:
	li	a2,4
	lla	a1,W
	lla	a0,A
	tail	dot8_t
	.size	main, .-main
	.section	.srodata,"a"
	.align	3
	.type	W, @object
	.size	W, 4
W:
	.base64	"AgQGCA=="
	.zero	4
	.type	A, @object
	.size	A, 4
A:
	.base64	"AQMFBw=="
	.ident	"GCC: (GNU) 16.2.0"
	.section	.note.GNU-stack,"",@progbits
