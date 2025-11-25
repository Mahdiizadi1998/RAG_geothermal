# NEW SUMMARY METHOD - TO BE INSERTED INTO APP.PY

    def _handle_summary(self, query: str) -> Tuple[str, str]:
        """
        Generate summary by retrieving 8 data types from database and chunks
        
        8 Data Types:
        1. General Data: Well Name, License, Well Type, Location, Coordinates, Operator, Rig, Target Formation
        2. Drilling Timeline: Spud Date, End of Operations, Total Days
        3. Depths: TD (mAH), TVD, Sidetrack Start Depth
        4. Casing & Tubulars: Type, OD, Weight, Grade, Connection, Pipe ID (Nominal + Drift), Top/Bottom Depths
        5. Cementing: Lead/Tail volumes, Densities, TOC
        6. Fluids: Hole Size, Fluid Type, Density Range
        7. Geology/Formations: Formation names, depths, lithology, notes
        8. Incidents: Gas peaks, stuck pipe, mud losses
        """
        well_name = self._extract_well_name_from_query(query)
        
        if not well_name:
            return "⚠️ Please specify a well name for summary", ""
        
        logger.info(f"📝 Generating summary for {well_name}")
        
        summary_parts = []
        summary_parts.append(f"# Well Summary: {well_name}\n")
        
        # 1. Get ALL complete tables from DATABASE
        tables = self.db.get_complete_tables(well_name)
        logger.info(f"Retrieved {len(tables)} tables from database")
        
        # Organize tables by type/content
        for table in tables:
            table_text = self._format_table_markdown(table)
            summary_parts.append(table_text)
        
        # 2. Get narrative data from SEMANTIC SEARCH
        searches = [
            (f"{well_name} general data operator rig location", "## General Information"),
            (f"{well_name} spud date completion timeline", "## Timeline"),
            (f"{well_name} total depth TD TVD", "## Depths"),
            (f"{well_name} geology formations lithology", "## Geology"),
            (f"{well_name} incidents problems stuck pipe gas", "## Incidents")
        ]
        
        for search_query, section_title in searches:
            chunks = self.rag.retrieve(search_query, top_k=3)
            if chunks:
                summary_parts.append(f"\n{section_title}\n")
                combined_text = "\n\n".join([c['text'][:300] for c in chunks])
                
                if self.llm_available:
                    # Use LLM to summarize
                    try:
                        section_summary = self.llm.generate_answer(
                            f"Summarize {section_title} for {well_name} in 2-3 sentences",
                            [{'text': combined_text}]
                        )
                        summary_parts.append(section_summary)
                    except:
                        summary_parts.append(combined_text[:500])
                else:
                    summary_parts.append(combined_text[:500])
        
        final_summary = "\n\n".join(summary_parts)
        debug_info = f"Generated from {len(tables)} tables and semantic search"
        
        return final_summary, debug_info
    
    def _format_table_markdown(self, table: Dict) -> str:
        """Convert table to markdown format"""
        md = f"\n### {table.get('table_reference', 'Table')} (Page {table['source_page']})\n\n"
        
        # Headers
        headers = table.get('headers', [])
        if headers:
            md += "| " + " | ".join(str(h) for h in headers) + " |\n"
            md += "| " + " | ".join(["---"] * len(headers)) + " |\n"
        
        # Rows (limit to 20 rows)
        rows = table.get('rows', [])
        for row in rows[:20]:
            md += "| " + " | ".join(str(cell) for cell in row) + " |\n"
        
        if len(rows) > 20:
            md += f"\n*({len(rows) - 20} more rows...)*\n"
        
        return md
