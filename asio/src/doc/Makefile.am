if HAVE_DOXYGEN

noinst_DATA = ../../doc/html/index.html

doc: delete_doc ../../doc/html/index.html

delete_doc:
	-rm -f ../../doc/html/index.html
  
../../doc/html/index.html: asio.dox.in
	doxygen asio.dox

EXTRA_DIST = \
	../../doc/html/index.html \
	../../doc \
	asio_header.htm \
	asio_footer.htm

MAINTAINERCLEANFILES = \
	../../doc/html/*.html \
	../../doc/html/*.gif \
	../../doc/html/*.png \
	../../doc/html/*.css \
	../../doc/html/*.dot \
	Makefile.in

endif
