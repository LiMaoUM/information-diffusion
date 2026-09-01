# Build the response letter. Always formats before rendering, because editing a
# grid table cell shifts the pipes and pandoc will not parse it misaligned.
#
#   make            format and render
#   make check      report misaligned lines without writing

response_letter.pdf: response_letter.md format_response_letter.py
	python3 format_response_letter.py
	pandoc response_letter.md -o response_letter.pdf --pdf-engine=xelatex
	@pdfinfo response_letter.pdf | grep Pages

.PHONY: check
check:
	@python3 -c "import sys;\
	l=open('response_letter.md').read().split(chr(10));\
	s=next(i for i,x in enumerate(l) if x.startswith('+') and set(x)<=set('+-='));\
	c=[i for i,x in enumerate(l[s]) if x=='+'];\
	b=[i+1 for i,x in enumerate(l) if x.startswith('|') and [j for j,y in enumerate(x) if y=='|']!=c];\
	print('misaligned lines:', b or 'none')"
