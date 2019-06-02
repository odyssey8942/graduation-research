using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class DrawLine : MonoBehaviour {
    public GameObject car;
    public GameObject pillar;
    public LineRenderer Line ; 
    // Use this for initialization
    void Start () {
       
    }
	
	// Update is called once per frame
	void Update () {
        Line.SetPosition(0, car.transform.position);
        Line.SetPosition(1, pillar.transform.position);
    }
}
