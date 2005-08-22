if HAVE_DOXYGEN

noinst_DATA = ../../doc/html/asio.css ../../doc/html/index.html

doc: delete_doc ../../doc/html/asio.css ../../doc/html/index.html

delete_doc:
	-rm -f ../../doc/html/index.html
	-rm -f ../../doc/html/asio.css

../../doc/html/asio.css: asio.css
	mkdir -p ../../doc/html
	cp $(srcdir)/asio.css ../../doc/html/asio.css

../../doc/html/index.html: asio.dox.in
	doxygen asio.dox

EXTRA_DIST = \
	../../doc/html/index.html \
	../../doc \
	asio_header.htm \
	asio_footer.htm \
	asio.css \
	asio_dox.txt \
	release_checklist.htm \
	concepts/Async_Object.hpp \
	concepts/Async_Read_Stream.hpp \
	concepts/Async_Write_Stream.hpp \
	concepts/Dispatcher.hpp \
	concepts/Endpoint.hpp \
	concepts/Protocol.hpp \
	concepts/Stream.hpp \
	concepts/Sync_Read_Stream.hpp \
	concepts/Sync_Write_Stream.hpp \
	../examples/tutorial/daytime_dox.txt \
	../examples/tutorial/index_dox.txt \
	../examples/tutorial/timer_dox.txt

MAINTAINERCLEANFILES = \
	../../doc/html/*.html \
	../../doc/html/*.gif \
	../../doc/html/*.png \
	../../doc/html/*.css \
	../../doc/html/*.dot \
	../../doc/html/*.md5 \
	../../doc/html/*.map \
	Makefile.in

endif
