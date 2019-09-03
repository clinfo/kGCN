package org.gcnk.knime.nodes.CSVLabelExtractor;

import org.knime.core.node.NodeDialogPane;
import org.knime.core.node.NodeFactory;
import org.knime.core.node.NodeView;

/**
 * <code>NodeFactory</code> for the "CSVLabelExtractor" Node.
 * Read CSV file and extract labels.
 *
 * @author org.gcnk
 */
public class CSVLabelExtractorNodeFactory 
        extends NodeFactory<CSVLabelExtractorNodeModel> {

    /**
     * {@inheritDoc}
     */
    @Override
    public CSVLabelExtractorNodeModel createNodeModel() {
        return new CSVLabelExtractorNodeModel();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public int getNrNodeViews() {
        return 1;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public NodeView<CSVLabelExtractorNodeModel> createNodeView(final int viewIndex,
            final CSVLabelExtractorNodeModel nodeModel) {
        return new CSVLabelExtractorNodeView(nodeModel);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public boolean hasDialog() {
        return true;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public NodeDialogPane createNodeDialogPane() {
        return new CSVLabelExtractorNodeDialog();
    }

}

