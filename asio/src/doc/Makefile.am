if HAVE_DOXYGEN

noinst_DATA = $(top_builddir)/doc/html/asio.css $(top_builddir)/doc/html/index.html

doc: delete_doc $(top_builddir)/doc/html/asio.css $(top_builddir)/doc/html/index.html

delete_doc:
	-rm -f $(top_builddir)/doc/html/index.html
	-rm -f $(top_builddir)/doc/html/asio.css
  
$(top_builddir)/doc/html/asio.css: $(srcdir)/asio.css
	mkdir -p $(top_builddir)/doc/html
	cp $(srcdir)/asio.css $(top_builddir)/doc/html/asio.css

$(top_builddir)/doc/html/index.html: asio.dox
	doxygen asio.dox

EXTRA_DIST = \
	$(top_builddir)/doc/html/index.html \
	$(top_builddir)/doc \
	asio_header.htm \
	asio_footer.htm \
	asio.css \
	asio_dox.txt \
	release_checklist.htm \
  concepts/Async_Object.hpp \
  concepts/Async_Recv_Stream.hpp \
  concepts/Async_Send_Stream.hpp \
  concepts/Dispatcher.hpp \
  concepts/Endpoint.hpp \
  concepts/Protocol.hpp \
  concepts/Stream.hpp \
  concepts/Sync_Recv_Stream.hpp \
  concepts/Sync_Send_Stream.hpp \
	../examples/tutorial/daytime_dox.txt \
	../examples/tutorial/index_dox.txt \
	../examples/tutorial/timer_dox.txt

MAINTAINERCLEANFILES = \
	$(top_builddir)/doc/html/*.html \
	$(top_builddir)/doc/html/*.gif \
	$(top_builddir)/doc/html/*.png \
	$(top_builddir)/doc/html/*.css \
	$(top_builddir)/doc/html/*.dot \
	$(top_builddir)/doc/html/*.md5 \
	$(top_builddir)/doc/html/*.map \
	$(srcdir)/Makefile.in

endif
