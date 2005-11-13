if HAVE_DOXYGEN

noinst_DATA = \
	../../doc/index.html \
	../../doc/building.html \
	../../doc/asio.css \
	../../doc/reference/asio.css \
	../../doc/reference/index.html \
	../../doc/design/asio.css \
	../../doc/design/index.html \
	../../doc/examples/asio.css \
	../../doc/examples/index.html \
	../../doc/tutorial/asio.css \
	../../doc/tutorial/index.html

doc: \
	delete_doc \
	../../doc/index.html \
	../../doc/building.html \
	../../doc/asio.css \
	../../doc/reference/asio.css \
	../../doc/reference/index.html \
	../../doc/design/asio.css \
	../../doc/design/index.html \
	../../doc/examples/asio.css \
	../../doc/examples/index.html \
	../../doc/tutorial/asio.css \
	../../doc/tutorial/index.html

delete_doc:
	-rm -rf ../../doc

../../doc/index.html: index.html.in
	mkdir -p ../../doc
	cp $(srcdir)/index.html ../../doc/index.html

../../doc/building.html: building.html.in
	mkdir -p ../../doc
	cp $(srcdir)/building.html ../../doc/building.html

../../doc/asio.css: asio.css
	mkdir -p ../../doc
	cp $(srcdir)/asio.css ../../doc/asio.css

../../doc/reference/asio.css: asio.css
	mkdir -p ../../doc/reference
	cp $(srcdir)/asio.css ../../doc/reference/asio.css

../../doc/reference/index.html: reference.dox.in
	mkdir -p ../../doc/reference
	doxygen reference.dox

../../doc/design/asio.css: asio.css
	mkdir -p ../../doc/design
	cp $(srcdir)/asio.css ../../doc/design/asio.css

../../doc/design/index.html: design.dox.in
	mkdir -p ../../doc/design
	doxygen design.dox

../../doc/examples/asio.css: asio.css
	mkdir -p ../../doc/examples
	cp $(srcdir)/asio.css ../../doc/examples/asio.css

../../doc/examples/index.html: examples.dox.in
	mkdir -p ../../doc/examples
	doxygen examples.dox

../../doc/tutorial/asio.css: asio.css
	mkdir -p ../../doc/tutorial
	cp $(srcdir)/asio.css ../../doc/tutorial/asio.css

../../doc/tutorial/index.html: tutorial.dox.in
	mkdir -p ../../doc/tutorial
	doxygen tutorial.dox

EXTRA_DIST = \
	../../doc/index.html \
	../../doc/building.html \
	../../doc/asio.css \
	../../doc/reference/index.html \
	../../doc/design/index.html \
	../../doc/examples/index.html \
	../../doc/tutorial/index.html \
	../../doc \
	asio.css \
	design_header.htm \
	design_footer.htm \
	examples_dox.txt \
	examples_header.htm \
	examples_footer.htm \
	reference_dox.txt \
	reference_header.htm \
	reference_footer.htm \
	release_checklist.htm \
	tutorial_header.htm \
	tutorial_footer.htm \
	concepts/Async_Object.hpp \
	concepts/Async_Read_Stream.hpp \
	concepts/Async_Write_Stream.hpp \
	concepts/Const_Buffers.hpp \
	concepts/Dispatcher.hpp \
	concepts/Endpoint.hpp \
	concepts/Error_Source.hpp \
	concepts/IO_Control_Command.hpp \
	concepts/Mutable_Buffers.hpp \
	concepts/Protocol.hpp \
	concepts/Socket_Option.hpp \
	concepts/Stream.hpp \
	concepts/Sync_Read_Stream.hpp \
	concepts/Sync_Write_Stream.hpp \
	design/closeascancel_dox.txt \
	design/handlers_dox.txt \
	design/index_dox.txt \
	design/services_dox.txt \
	../examples/tutorial/daytime_dox.txt \
	../examples/tutorial/index_dox.txt \
	../examples/tutorial/timer_dox.txt

MAINTAINERCLEANFILES = \
	../../doc/*.html \
	../../doc/*.css \
	../../doc/reference/*.html \
	../../doc/reference/*.gif \
	../../doc/reference/*.png \
	../../doc/reference/*.css \
	../../doc/reference/*.dot \
	../../doc/reference/*.md5 \
	../../doc/reference/*.map \
	../../doc/examples/*.html \
	../../doc/examples/*.gif \
	../../doc/examples/*.png \
	../../doc/examples/*.css \
	../../doc/examples/*.dot \
	../../doc/examples/*.md5 \
	../../doc/examples/*.map \
	../../doc/tutorial/*.html \
	../../doc/tutorial/*.gif \
	../../doc/tutorial/*.png \
	../../doc/tutorial/*.css \
	../../doc/tutorial/*.dot \
	../../doc/tutorial/*.md5 \
	../../doc/tutorial/*.map \
	Makefile.in

endif
