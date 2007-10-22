<?xml version="1.0" encoding="utf-8"?>
<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="1.0">

<!--
  Copyright (c) 2003-2007 Christopher M. Kohlhoff (chris at kohlhoff dot com)

  Distributed under the Boost Software License, Version 1.0. (See accompanying
  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
-->

<xsl:output method="text"/>
<xsl:strip-space elements="*"/>


<xsl:variable name="newline">
<xsl:text>
</xsl:text>
</xsl:variable>


<xsl:template match="/doxygen">
<xsl:text>[/
 / Copyright (c) 2003-2007 Christopher M. Kohlhoff (chris at kohlhoff dot com)
 /
 / Distributed under the Boost Software License, Version 1.0. (See accompanying
 / file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 /]

[section:tutorial Tutorial]

</xsl:text>

  <xsl:for-each select="compounddef[@kind = 'page' and @id = 'indexpage']">
    <xsl:apply-templates select="detaileddescription"/>
  </xsl:for-each>

  <xsl:for-each select="
      compounddef[
        @kind = 'page' and
        @id != 'indexpage' and
        not(contains(@id, 'src'))]">
    <xsl:text>[section:</xsl:text>
    <xsl:value-of select="concat(@id, ' ', title)"/>
    <xsl:text>]</xsl:text>
    <xsl:value-of select="$newline"/>
    <xsl:value-of select="$newline"/>
    <xsl:apply-templates select="detaileddescription"/>
    <xsl:variable name="srcid" select="concat(@id, 'src')"/>
    <xsl:if test="count(/doxygen/compounddef[@id = $srcid]) &gt; 0">
      <xsl:value-of select="$newline"/>
      <xsl:value-of select="$newline"/>
      <xsl:text>[section:src </xsl:text>
      <xsl:value-of select="/doxygen/compounddef[@id = $srcid]/title"/>
      <xsl:text>]</xsl:text>
      <xsl:value-of select="$newline"/>
      <xsl:value-of select="$newline"/>
      <xsl:apply-templates select="/doxygen/compounddef[@id = $srcid]/detaileddescription"/>
      <xsl:text>[endsect]</xsl:text>
      <xsl:value-of select="$newline"/>
      <xsl:value-of select="$newline"/>
    </xsl:if>
    <xsl:text>[endsect]</xsl:text>
    <xsl:value-of select="$newline"/>
    <xsl:value-of select="$newline"/>
  </xsl:for-each>

  <xsl:value-of select="$newline"/>
  <xsl:text>[endsect]</xsl:text>

</xsl:template>


<!--========== Markup ==========-->

<xsl:template match="para">
  <xsl:apply-templates/>
  <xsl:value-of select="$newline"/>
  <xsl:value-of select="$newline"/>
</xsl:template>


<xsl:template match="title">
  <xsl:variable name="title">
    <xsl:value-of select="."/>
  </xsl:variable>
  <xsl:if test="string-length($title) > 0">
    <xsl:text>[heading </xsl:text>
    <xsl:value-of select="."/>
    <xsl:text>]</xsl:text>
    <xsl:value-of select="$newline"/>
    <xsl:value-of select="$newline"/>
  </xsl:if>
</xsl:template>


<xsl:template match="programlisting">
  <xsl:value-of select="$newline"/>
  <xsl:apply-templates/>
  <xsl:value-of select="$newline"/>
</xsl:template>


<xsl:template match="codeline">
  <xsl:if test="string-length(.) &gt; 0">
    <xsl:text>  ``''''''``</xsl:text>
  </xsl:if>
  <xsl:apply-templates mode="codeline"/>
  <xsl:value-of select="$newline"/>
</xsl:template>


<xsl:template match="sp">
  <xsl:text> </xsl:text>
</xsl:template>


<xsl:template match="sp" mode="codeline">
  <xsl:text> </xsl:text>
</xsl:template>


<xsl:template match="linebreak">
  <xsl:value-of select="$newline"/>
  <xsl:value-of select="$newline"/>
</xsl:template>


<xsl:template match="listitem">
  <xsl:text>* </xsl:text>
  <xsl:apply-templates/>
  <xsl:value-of select="$newline"/>
</xsl:template>


<xsl:template match="emphasis">
  <xsl:text>[*</xsl:text>
  <xsl:value-of select="."/>
  <xsl:text>]</xsl:text>
</xsl:template>


<xsl:template match="computeroutput">
  <xsl:text>`</xsl:text>
  <xsl:value-of select="."/>
  <xsl:text>`</xsl:text>
</xsl:template>


<xsl:template match="text()">
  <xsl:variable name="text" select="."/>
  <xsl:variable name="starts-with-whitespace" select="
      starts-with($text, ' ') or starts-with($text, $newline)"/>
  <xsl:variable name="preceding-node-name">
    <xsl:for-each select="preceding-sibling::*">
      <xsl:if test="position() = last()">
        <xsl:value-of select="local-name()"/>
      </xsl:if>
    </xsl:for-each>
  </xsl:variable>
  <xsl:variable name="after-newline" select="
      $preceding-node-name = 'programlisting' or
      $preceding-node-name = 'linebreak'"/>
  <xsl:choose>
    <xsl:when test="$starts-with-whitespace and $after-newline">
      <xsl:call-template name="escape-text">
        <xsl:with-param name="text">
          <xsl:call-template name="strip-leading-whitespace">
            <xsl:with-param name="text" select="$text"/>
          </xsl:call-template>
        </xsl:with-param>
      </xsl:call-template>
    </xsl:when>
    <xsl:otherwise>
      <xsl:call-template name="escape-text">
        <xsl:with-param name="text" select="$text"/>
      </xsl:call-template>
    </xsl:otherwise>
  </xsl:choose>
</xsl:template>


<xsl:template name="strip-leading-whitespace">
  <xsl:param name="text"/>
  <xsl:choose>
    <xsl:when test="starts-with($text, ' ')">
      <xsl:call-template name="strip-leading-whitespace">
        <xsl:with-param name="text" select="substring($text, 2)"/>
      </xsl:call-template>
    </xsl:when>
    <xsl:when test="starts-with($text, $newline)">
      <xsl:call-template name="strip-leading-whitespace">
        <xsl:with-param name="text" select="substring($text, 2)"/>
      </xsl:call-template>
    </xsl:when>
    <xsl:otherwise>
      <xsl:value-of select="$text"/>
    </xsl:otherwise>
  </xsl:choose>
</xsl:template>


<xsl:template name="escape-text">
  <xsl:param name="text"/>
  <xsl:choose>
    <xsl:when test="contains($text, '_')">
      <xsl:value-of select="substring-before($text, '_')"/>
      <xsl:text>\_</xsl:text>
      <xsl:call-template name="escape-text">
        <xsl:with-param name="text" select="substring-after($text, '_')"/>
      </xsl:call-template>
    </xsl:when>
    <xsl:otherwise>
      <xsl:value-of select="$text"/>
    </xsl:otherwise>
  </xsl:choose>
</xsl:template>


<xsl:template match="ref[@kindref='compound']">
  <xsl:variable name="refid">
    <xsl:value-of select="@refid"/>
  </xsl:variable>
  <xsl:choose>
    <xsl:when test="count(/doxygen/compounddef[@id=$refid]) &gt; 0">
      <xsl:text>[link asio.tutorial.</xsl:text>
      <xsl:choose>
        <xsl:when test="contains($refid, 'src')">
          <xsl:value-of select="substring-before($refid, 'src')"/>
          <xsl:text>.src</xsl:text>
        </xsl:when>
        <xsl:otherwise>
          <xsl:value-of select="$refid"/>
        </xsl:otherwise>
      </xsl:choose>
      <xsl:value-of select="concat(' ', .)"/>
      <xsl:text>]</xsl:text>
    </xsl:when>
    <xsl:otherwise>
      <xsl:apply-templates/>
    </xsl:otherwise>
  </xsl:choose>
</xsl:template>


<xsl:template match="ref[@kindref='member']">
  <xsl:variable name="refid">
    <xsl:value-of select="@refid"/>
  </xsl:variable>
  <xsl:choose>
    <xsl:when test="$refid='index_1index'">
      <xsl:text>[link asio.tutorial </xsl:text>
      <xsl:value-of select="."/>
      <xsl:text>]</xsl:text>
    </xsl:when>
    <xsl:otherwise>
      <xsl:apply-templates/>
    </xsl:otherwise>
  </xsl:choose>
</xsl:template>


<xsl:template match="ulink">
  <xsl:text>[@</xsl:text>
  <xsl:value-of select="@url"/>
  <xsl:text> </xsl:text>
  <xsl:value-of select="."/>
  <xsl:text>]</xsl:text>
</xsl:template>


</xsl:stylesheet>
