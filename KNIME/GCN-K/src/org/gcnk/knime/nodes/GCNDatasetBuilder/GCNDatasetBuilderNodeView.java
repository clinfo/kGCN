package org.gcnk.knime.nodes.GCNDatasetBuilder;

import org.knime.core.node.NodeView;

/**
 * <code>NodeView</code> for the "GCNDatasetBuilder" Node.
 * Read labels from CSV file and Create GCN Dataset together with graph structure and atom features.
 *
 * @author org.gcnk
 */
public class GCNDatasetBuilderNodeView extends NodeView<GCNDatasetBuilderNodeModel> {

    /**
     * Creates a new view.
     * 
     * @param nodeModel The model (class: {@link GCNDatasetBuilderNodeModel})
     */
    protected GCNDatasetBuilderNodeView(final GCNDatasetBuilderNodeModel nodeModel) {
        super(nodeModel);

        // TODO instantiate the components of the view here.

    }

    /**
     * {@inheritDoc}
     */
    @Override
    protected void modelChanged() {

        // TODO retrieve the new model from your nodemodel and 
        // update the view.
        GCNDatasetBuilderNodeModel nodeModel = 
            (GCNDatasetBuilderNodeModel)getNodeModel();
        assert nodeModel != null;
        
        // be aware of a possibly not executed nodeModel! The data you retrieve
        // from your nodemodel could be null, emtpy, or invalid in any kind.
        
    }

    /**
     * {@inheritDoc}
     */
    @Override
    protected void onClose() {
    
        // TODO things to do when closing the view
    }

    /**
     * {@inheritDoc}
     */
    @Override
    protected void onOpen() {

        // TODO things to do when opening the view
    }

}

