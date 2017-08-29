start = 190
end = 202

file_f = open('list_f.txt', 'wt')
file_p = open('list_p.txt', 'wt')

for i in xrange(start, end):
	file_f.write('00%d.png\n'%i)
	file_p.write('00%d_pred.png\n'%i)

